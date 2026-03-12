"""
inference.py
Inference mode - sample a random image from the test/val dataset,
run a forward pass through the trained model, show top-k predictions.

No file uploading — samples come directly from the Data Prep graph
using train=False on the dataset node.
"""

from __future__ import annotations
import pathlib
import dearpygui.dearpygui as dpg
from ui.console import log


# Texture

_TEX_TAG = "inf_preview_tex"
_PREV_W  = 224
_PREV_H  = 224


def _ensure_texture() -> None:
    if dpg.does_item_exist(_TEX_TAG):
        return
    blank = [0.1, 0.1, 0.1, 1.0] * (_PREV_W * _PREV_H)
    with dpg.texture_registry():
        dpg.add_dynamic_texture(_PREV_W, _PREV_H, blank, tag=_TEX_TAG)


def _pil_to_texture(pil_img) -> None:
    import numpy as np
    img  = pil_img.convert("RGBA").resize((_PREV_W, _PREV_H))
    data = (np.array(img, dtype=np.float32) / 255.0).flatten().tolist()
    dpg.set_value(_TEX_TAG, data)


# State

_state: dict = {
    "ckpt_path":   "",
    "last_sample": None,
}


# Window

def open_inference_window() -> None:
    tag = "inference_window"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    _ensure_texture()

    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    ww, wh = 520, 500
    px = (vw - ww) // 2
    py = (vh - wh) // 2

    with dpg.window(label="Inference", tag=tag, width=ww, height=wh,
                    pos=(px, py), no_collapse=True, modal=False):

        dpg.add_text("Samples random images from the test set and runs your model.",
                     color=(160, 160, 160))
        dpg.add_spacer(height=8)

        dpg.add_text("Checkpoint (.pth)", color=(200, 200, 200))
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="inf_ckpt_path", width=350,
                               hint="path/to/checkpoint.pth",
                               default_value=_state["ckpt_path"])
            dpg.add_button(label="Browse", callback=_browse_checkpoint)

        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_text("Top-k:", color=(180, 180, 180))
            dpg.add_slider_int(tag="inf_topk", default_value=5,
                               min_value=1, max_value=10, width=120)

        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_button(label="New Sample", width=140, callback=_new_sample)
            with dpg.theme() as blue_th:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button,        (40, 100, 160, 220))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 130, 200, 255))
            dpg.bind_item_theme(dpg.last_item(), blue_th)

            dpg.add_spacer(width=8)

            dpg.add_button(label="Run Inference", width=140, callback=_run_on_current_sample)
            _apply_green_theme(dpg.last_item())

        dpg.add_spacer(height=4)
        dpg.add_text("", tag="inf_status", color=(140, 140, 140))
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Sample", color=(140, 140, 140))
                dpg.add_image(_TEX_TAG, width=_PREV_W, height=_PREV_H)
                dpg.add_text("", tag="inf_true_label", color=(120, 180, 120))

            dpg.add_spacer(width=16)

            with dpg.group():
                dpg.add_text("Predictions", color=(140, 140, 140))
                with dpg.child_window(tag="inf_results", width=220,
                                      height=_PREV_H, border=False):
                    dpg.add_text("Press 'Sample + Run' to start.",
                                 color=(100, 100, 100))


def _apply_green_theme(item) -> None:
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (40, 120, 60, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 80, 255))
    dpg.bind_item_theme(item, th)


def _browse_checkpoint() -> None:
    def _cb(sender, app_data):
        path = app_data.get("file_path_name", "")
        if path and dpg.does_item_exist("inf_ckpt_path"):
            dpg.set_value("inf_ckpt_path", path)
            _state["ckpt_path"] = path

    tag = "inf_ckpt_dialog"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.file_dialog(label="Select checkpoint", tag=tag,
                         width=600, height=400, callback=_cb, modal=True):
        dpg.add_file_extension(".pth")
        dpg.add_file_extension(".pt")


# Dataset sampling

def _load_test_dataset():
    """
    Load the test dataset from the Data Prep graph.
    Prefers the DataLoader (val) chain with its own dataset node.
    Falls back to the train dataset node forced to train=False.
    Returns (dataset, error_string).
    """
    from torchvision import datasets, transforms
    from engine.graph import (topological_sort, get_tab_by_role,
                               _DATASET_BLOCKS, _AUG_BLOCKS, build_graph)

    tab = get_tab_by_role("data_prep")
    if tab is None:
        return None, "No Data Prep tab found."

    try:
        ordered = topological_sort(tab)
    except Exception as e:
        return None, f"Graph error: {e}"

    graph = build_graph(tab)
    nodes = list(graph.values())

    TORCHVISION_DATASETS = {
        "MNIST":        datasets.MNIST,
        "CIFAR10":      datasets.CIFAR10,
        "CIFAR100":     datasets.CIFAR100,
        "FashionMNIST": datasets.FashionMNIST,
    }

    def _ancestors_of(loader_node):
        if loader_node is None:
            return []
        targets = {loader_node.ntag}
        found   = set()
        changed = True
        while changed:
            changed = False
            for _, (a1, a2) in tab["links"].items():
                if isinstance(a1, int): a1 = dpg.get_item_alias(a1) or str(a1)
                if isinstance(a2, int): a2 = dpg.get_item_alias(a2) or str(a2)
                dp = a2.split("_"); sp = a1.split("_")
                if len(dp) >= 3 and len(sp) >= 3:
                    d = f"node_{dp[1]}_{dp[2]}"
                    s = f"node_{sp[1]}_{sp[2]}"
                    if d in targets and s not in found:
                        found.add(s); targets.add(s); changed = True
        return [n for n in ordered if n.ntag in found]

    def _inference_transform(chain_nodes):
        tlist = []
        for n in chain_nodes:
            label = n.block_label
            if label not in _AUG_BLOCKS:
                continue
            if label == "ToTensor":
                tlist.append(transforms.ToTensor())
            elif label == "Resize":
                tlist.append(transforms.Resize(int(n.params.get("size","224") or "224")))
            elif label == "CenterCrop":
                tlist.append(transforms.CenterCrop(int(n.params.get("size","224") or "224")))
            elif label == "RandomCrop":
                tlist.append(transforms.CenterCrop(int(n.params.get("size","32") or "32")))
            elif label == "Normalize":
                mean = n.params.get("mean","0.5").strip() or "0.5"
                std  = n.params.get("std", "0.5").strip() or "0.5"
                try:
                    mean = eval(mean) if "[" in mean or "(" in mean else [float(mean)]*3
                    std  = eval(std)  if "[" in std  or "(" in std  else [float(std)] *3
                except Exception:
                    mean, std = [0.5,0.5,0.5],[0.5,0.5,0.5]
                tlist.append(transforms.Normalize(mean=mean, std=std))
            elif label == "Grayscale":
                tlist.append(transforms.Grayscale(int(n.params.get("num_output_channels","1") or "1")))
        if not any(isinstance(t, transforms.ToTensor) for t in tlist):
            tlist.insert(0, transforms.ToTensor())
        return transforms.Compose(tlist)

    # Prefer val chain, fall back to any chain
    val_node   = next((n for n in nodes if n.block_label == "DataLoader (val)"),   None)
    train_node = next((n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")), None)

    chain  = _ancestors_of(val_node) or _ancestors_of(train_node) or list(ordered)
    ds_node = next((n for n in chain if n.block_label in _DATASET_BLOCKS), None)
    if ds_node is None:
        ds_node = next((n for n in ordered if n.block_label in _DATASET_BLOCKS), None)
    if ds_node is None:
        return None, "No Dataset node found in Data Prep tab."

    transform = _inference_transform(chain)
    label     = ds_node.block_label
    root      = ds_node.params.get("root","./data").strip() or "./data"
    download  = ds_node.params.get("download","True").strip()

    # Val chain dataset node likely has train=False already set.
    # If using the train chain as fallback, force train=False for test sampling.
    train_param = ds_node.params.get("train","True").strip().lower()
    use_train   = train_param != "false"
    # If this is the only dataset node and no val chain exists, use test split
    if val_node is None:
        use_train = False

    try:
        if label in TORCHVISION_DATASETS:
            ds = TORCHVISION_DATASETS[label](
                root=root,
                train=use_train,
                download=(download.lower() != "false"),
                transform=transform,
            )
            return ds, None
        elif label == "ImageFolder":
            if not pathlib.Path(root).exists():
                return None, f"ImageFolder root not found: {root}"
            from torchvision.datasets import ImageFolder
            return ImageFolder(root=root, transform=transform), None
        else:
            return None, f"Dataset '{label}' not supported for inference sampling."
    except Exception as e:
        return None, str(e)


def _get_random_sample():
    """Return (pil_img_for_display, tensor_for_model, true_label) or raise."""
    import random
    import torch
    from torchvision import transforms as T

    ds, err = _load_test_dataset()
    if ds is None:
        raise ValueError(err or "Could not load dataset.")

    idx            = random.randint(0, len(ds) - 1)
    tensor, label  = ds[idx]

    # Build a display image from the tensor (clamp, un-normalise roughly)
    t_display = tensor.clone().detach()
    t_display = torch.clamp(t_display, 0.0, 1.0)
    to_pil    = T.ToPILImage()
    pil_img   = to_pil(t_display)

    return pil_img, tensor, label


# Actions

def _new_sample() -> None:
    _set_status("Loading sample...")
    _clear_results()
    try:
        pil_img, tensor, true_label = _get_random_sample()
        _state["last_sample"] = (pil_img, tensor, true_label)
        _pil_to_texture(pil_img)
        if dpg.does_item_exist("inf_true_label"):
            dpg.set_value("inf_true_label", f"True class: {true_label}")
        _set_status("Sample loaded. Press 'Run Inference' to classify.")
    except Exception as e:
        _set_status(f"Error: {e}", error=True)
        log(f"Inference sample error: {e}", "error")


def _run_on_current_sample() -> None:
    """Run inference on the currently loaded sample without fetching a new one."""
    if _state["last_sample"] is None:
        # No sample yet — load one first automatically
        _new_sample()
        if _state["last_sample"] is None:
            return

    ckpt_path = dpg.get_value("inf_ckpt_path").strip() if dpg.does_item_exist("inf_ckpt_path") else ""
    topk      = dpg.get_value("inf_topk")               if dpg.does_item_exist("inf_topk")      else 5

    if not ckpt_path:
        _set_status("Select a checkpoint first.", error=True)
        return
    if not pathlib.Path(ckpt_path).exists():
        _set_status("Checkpoint file not found.", error=True)
        return

    try:
        import torch
    except ImportError:
        log("Inference: PyTorch not installed.", "error")
        return

    _set_status("Running model...")
    _clear_results()

    try:
        _, tensor, true_label = _state["last_sample"]

        from engine.run import _build_torch_model, _resolve_device
        device = _resolve_device("auto")
        model  = _build_torch_model(device)
        model.eval()

        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd)

        inp = tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)[0]

        topk_vals, topk_idxs = torch.topk(probs, min(topk, probs.shape[0]))
        correct = int(true_label) if not isinstance(true_label, int) else true_label

        for rank, (idx, conf) in enumerate(zip(topk_idxs.tolist(), topk_vals.tolist()), 1):
            _show_result_row(rank, idx, conf, is_correct=(idx == correct))

        top1    = topk_idxs[0].item()
        top1_c  = topk_vals[0].item()
        verdict = "CORRECT" if top1 == correct else "WRONG"
        _set_status(f"Top-1: class {top1} ({top1_c:.1%}) — {verdict}")
        log(f"Inference: pred={top1} conf={top1_c:.3f} true={correct} [{verdict}]", "success")

    except Exception as e:
        _clear_results()
        _show_result_text(f"Error: {e}", (220, 80, 80))
        _set_status(f"Model error: {e}", error=True)
        log(f"Inference error: {e}", "error")


def _sample_and_run() -> None:
    """Load a new sample and immediately run inference on it."""
    _new_sample()
    _run_on_current_sample()


# ── UI helpers ────────────────────────────────────────────

def _clear_results() -> None:
    if dpg.does_item_exist("inf_results"):
        dpg.delete_item("inf_results", children_only=True)


def _show_result_text(msg: str, color=(160, 160, 160)) -> None:
    if dpg.does_item_exist("inf_results"):
        dpg.add_text(msg, color=color, parent="inf_results")


def _show_result_row(rank: int, class_idx: int, conf: float,
                     is_correct: bool = False) -> None:
    if not dpg.does_item_exist("inf_results"):
        return
    pct = conf * 100
    col = (80, 220, 120) if is_correct else (220, 180, 60) if conf > 0.3 else (180, 180, 180)
    marker = "  v" if is_correct else ""
    dpg.add_text(f"#{rank}  class {class_idx:>4}   {pct:5.1f}%{marker}",
                 color=col, parent="inf_results")
    dpg.add_progress_bar(default_value=conf, width=-1, overlay="", parent="inf_results")
    dpg.add_spacer(height=4, parent="inf_results")


def _set_status(msg: str, error: bool = False) -> None:
    if dpg.does_item_exist("inf_status"):
        col = (220, 80, 80) if error else (140, 140, 140)
        dpg.set_value("inf_status", msg)
        dpg.configure_item("inf_status", color=col)