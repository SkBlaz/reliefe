import reliefe
import scipy.io as sio

# Load the data first
mat_obj = sio.loadmat("../data/mlc/Science1.mat")
x = mat_obj['input_space']  ## scipy csr sparse matrix (or numpy dense)
y = mat_obj['target_space']  ## scipy csr sparse matrix (or numpy dense)

print(y.shape)  ## 40 possible labels!

reliefE_instance = reliefe.ReliefE()  # Initialize default ReliefE
reliefE_instance.fit(x, y)  # Compute rankings
print(reliefE_instance.feature_importances_
      )  # rankings for features (same order as x)
