import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely_integration.serve.stdcseg import STDCSeg


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


model = STDCSeg(
    sliding_window_mode="none",
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()
