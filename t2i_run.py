from waitress import serve
import t2i_API
import warnings
warnings.filterwarnings("ignore")

serve(t2i_API.app, host='0.0.0.0', port=5023)