from waitress import serve
import shape_API
import warnings
warnings.filterwarnings("ignore")

serve(shape_API.app, host='0.0.0.0',port=5013)