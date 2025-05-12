import os

import nncf
import openvino
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

DATASET_DIR = "./coco/val2017"
DET_MODEL_NAME = "yolo11s"
OV_MODEL_NAME = "yolo11s_openvino_model"
INT8_MODEL_DET_PATH = "./yolo11s_int8_openvino_model/yolo11s_int8.xml"


class CocoValDatasetFlat(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
)

dataset = CocoValDatasetFlat(DATASET_DIR, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=1,
)


def transform_fn(data_item):
    return data_item


def main():
    model = YOLO(f"{DET_MODEL_NAME}.pt")
    model.export(format="openvino", dynamic=True)

    # setup openvino
    core = openvino.Core()
    ov_model = core.read_model(f"./{OV_MODEL_NAME}/{DET_MODEL_NAME}.xml")
    # ov_compiled_model = core.compile_model(ov_model, device_name="CPU")

    # setup quantization
    quantization_dataset = nncf.Dataset(dataloader, transform_fn)
    ignored_scope = nncf.IgnoredScope(  # post-processing
        subgraphs=[
            nncf.Subgraph(
                inputs=[
                    f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat",
                    f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_1",
                    f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_2",
                ],
                outputs=[f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_7"],
            )
        ]
    )

    # Quantize and save model
    quantized_model = nncf.quantize(
        ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED, ignored_scope=ignored_scope
    )
    openvino.save_model(quantized_model, INT8_MODEL_DET_PATH)


if __name__ == "__main__":
    main()
