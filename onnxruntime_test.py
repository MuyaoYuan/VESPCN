import onnx
import onnxruntime
from PIL import Image
from datasetProcess.SRtransforms import *

img_path = '003.jpg'
img = Image.open(img_path)
input_tensor = ToTensorWithTranspose()(img)
input_tensor = input_tensor.view(1, *input_tensor.size())
input_np = input_tensor.detach().cpu().numpy()
print(input_np.shape)

onnxpath = 'ESPCN_ONNX.onnx'
onnx_model = onnx.load(onnxpath)
check = onnx.checker.check_model(onnx_model)
print('check: ', check)

ort_session = onnxruntime.InferenceSession(onnxpath)
ort_outs = ort_session.run(None, {'input': input_np})
np_out = ort_outs[0][0].transpose(1,2,0)

img_out = Image.fromarray(np.uint8(np_out))
savepath = 'test' + img_path 
img_out.save(savepath)