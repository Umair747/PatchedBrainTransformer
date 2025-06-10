import argparse
import torch
from src.utils import SeqDataset, get_BNCI2014001, get_BNCI2014004, zero_mean_unit_var
from src.model import PBT


def load_dataset(config):
    train_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=False,
    )
    test_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=False,
    )

    if config["data_set"] == "BNCI2014001":
        data, labels, meta, channels = get_BNCI2014001(
            subject=list(range(1, 10)),
            freq_min=config["freq"][0],
            freq_max=config["freq"][1],
        )
        train_data = data[meta["session"] == "session_T"]
        train_labels = labels[meta["session"] == "session_T"]
        train_meta = meta.loc[meta["session"] == "session_T"]
        test_data = data[meta["session"] == "session_E"]
        test_labels = labels[meta["session"] == "session_E"]
        test_meta = meta.loc[meta["session"] == "session_E"]
    elif config["data_set"] == "BNCI2014004":
        data, labels, meta, channels = get_BNCI2014004(
            subject=list(range(1, 10)),
            freq_min=config["freq"][0],
            freq_max=config["freq"][1],
        )
        train_idx = (meta["session"].isin(["session_0", "session_1", "session_2"]))
        test_idx = (meta["session"].isin(["session_3", "session_4"]))
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        train_meta = meta.loc[train_idx]
        test_data = data[test_idx]
        test_labels = labels[test_idx]
        test_meta = meta.loc[test_idx]
    else:
        raise ValueError("Unknown data_set")

    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)

    train_set.append_data_set(train_data, channel_names=channels, label=train_labels)
    test_set.append_data_set(test_data, channel_names=channels, label=test_labels)

    train_set.prepare_data_set()
    test_set.prepare_data_set(train_set.dict_channels)
    return train_set, test_set, len(set(test_labels))


def evaluate(checkpoint_path):
    chk = torch.load(checkpoint_path, map_location="cpu")
    config = chk.get("config", {})
    if not config:
        raise RuntimeError("No config stored in checkpoint")

    train_set, test_set, n_classes = load_dataset(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PBT(
        d_input=config["d_input"],
        n_classes=n_classes,
        num_embeddings=max(torch.cat(list(train_set.dict_channels.values()))).item() + 1,
        num_tokens_per_channel=config["num_tokens_per_channel"],
        d_model=config["d_model"],
        n_blocks=config["num_transformer_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        device=device,
        learnable_cls=config.get("learnable_cls", False),
        bias_transformer=config.get("bias_transformer", True),
        bert=config.get("bert_supervised", False) or config.get("pre_train_bert", False),
    )
    model.load_state_dict(chk["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.get("batch_size", 64),
        shuffle=False,
        drop_last=False,
        collate_fn=test_set.my_collate,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            logits = torch.empty(0, n_classes).to(device)
            labels = torch.empty(0, dtype=torch.long).to(device)
            for sub_batch in range(len(data["patched_eeg_token"])):
                _, out, _ = model(
                    x=data["patched_eeg_token"][sub_batch].to(device),
                    pos=data["pos_as_int"][sub_batch].long().to(device),
                )
                logits = torch.cat((logits, out), dim=0)
                labels = torch.cat((labels, data["labels"][sub_batch].long().to(device)), 0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PBT model accuracy")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.checkpoint)
