from typing import Dict, List

import supervisely as sly
from supervisely.app.exceptions import DialogWindowError
from supervisely.nn.inference import ModelSource
from supervisely.nn.prediction_dto import PredictionSegmentation

from supervisely_integration.serve.stdcnet_infer import (
    ModelConfigError,
    STDCNetSegmenter,
    load_model_config,
)


SERVE_PATH = "supervisely_integration/serve"
DEFAULT_MODEL_CONFIG = "supervisely_integration/model_config.yaml"

class STDCSeg(sly.nn.inference.SemanticSegmentation):
    FRAMEWORK_NAME = "STDC-Seg"
    MODELS = "supervisely_integration/models.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.yaml"

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        if model_source == ModelSource.PRETRAINED:
            checkpoint_path = model_files["checkpoint"]
            config_path = DEFAULT_MODEL_CONFIG
            model_meta = model_info["meta"]
            backbone = model_meta["backbone"]
            input_size = model_meta["input_size"]

        elif model_source == ModelSource.EXTERNAL:
            checkpoint_path = model_files["checkpoint"]
            config_path = model_files.get("config")
            if not config_path:
                raise ValueError(
                    "Config file is required for external models. "
                    "Please provide a config file in the model files."
                )
            backbone = STDCNetSegmenter.BACKBONE_FROM_CHECKPOINT
        else:
            raise ValueError(f"Unsupported model source: {model_source}")

        config = load_model_config(config_path)
        categories = config.get("SEG_CATEGORIES")
        if not categories:
            raise ValueError("SEG_CATEGORIES is required in the model config.")
        classes = [
            name
            for name, _ in sorted(categories.items(), key=lambda item: item[1])
        ]
        colors = STDCNetSegmenter.build_palette(config, len(classes))
        project_meta = self.create_model_meta(classes, colors)

        if model_source == ModelSource.EXTERNAL:
            try:
                input_size = (
                    int(config["NET_INP_WIDTH"]),
                    int(config["NET_INP_HEIGHT"]),
                )
            except KeyError as error:
                raise ValueError(
                    "NET_INP_WIDTH and NET_INP_HEIGHT are required in the "
                    "config for external models."
                ) from error

        try:
            segmenter = STDCNetSegmenter(
                checkpoint_path,
                device=device,
                n_classes=len(classes),
                backbone=backbone,
                input_size=input_size,
                config_path=config_path,
            )
        except ModelConfigError as error:
            raise DialogWindowError(
                "Model configuration does not match checkpoint",
                str(error),
            ) from error

        self.segmenter = segmenter
        self.classes = classes
        self._model_meta = project_meta
        self._get_confidence_tag_meta()

        self.max_batch_size = 1

    def create_model_meta(self, classes: List[str], colors) -> sly.ProjectMeta:
        if len(classes) != len(colors):
            raise ValueError(
                "Classes and colors have different lengths: "
                f"{len(classes)} and {len(colors)}."
            )
        obj_classes = [
            sly.ObjClass(
                name,
                self._get_obj_class_shape(),
                color=[int(channel) for channel in color],
            )
            for name, color in zip(classes, colors)
        ]
        return sly.ProjectMeta(obj_classes=obj_classes)

    def predict(
        self, image_path: str, settings: Dict
    ) -> List[PredictionSegmentation]:
        mask = self.segmenter.predict(image_path)
        return [PredictionSegmentation(mask)]

    def get_info(self) -> dict:
        info = super().get_info()
        if getattr(self, "segmenter", None) is not None:
            info["backbone"] = self.segmenter.backbone
            info["input_size"] = self.segmenter.input_size
        return info

    def shutdown_model(self):
        self.segmenter = None
        self._model_meta = None
        self.classes = None
        super().shutdown_model()
