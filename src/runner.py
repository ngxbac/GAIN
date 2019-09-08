from typing import Mapping, Any
from catalyst.dl.runner import WandbRunner
from models import GAIN, GCAM


class ModelRunner(WandbRunner):
    def predict_batch(self, batch: Mapping[str, Any]):
        if isinstance(self.model, GAIN):
            output, output_am, heatmap = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
                "logits_am": output_am,
                "heatmap": heatmap
            }
        elif isinstance(self.model, GCAM):
            output, heatmap = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
                "heatmap": heatmap
            }
        else:
            output = self.model(batch["images"])
            return {
                "logits": output
            }
