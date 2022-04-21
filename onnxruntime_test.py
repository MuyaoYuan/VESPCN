from PIL import Image
from datasetProcess.SRtransforms import *
import onnxruntime
 
img = Image.open("001.jpg")
input_tensor = ToTensorWithTranspose()(img)
input_np = input_tensor.detach().cpu().numpy()

ort_session = onnxruntime.InferenceSession('ESPCN_ONNX.onnx')