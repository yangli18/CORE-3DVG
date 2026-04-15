#include "ml_nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ml_nms", &ml_nms, "multi-label non-maximum suppression");
}