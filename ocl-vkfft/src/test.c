#include "vkFFT.h"

VkFFTResult almost_initializeVkFFT(VkFFTApplication *app,
                                   VkFFTConfiguration configuration) {
  printf("[C] Context pointer : %p\n", configuration.context);
  printf("[C] Queue pointer : %p\n", configuration.commandQueue);
  printf("[C] Device pointer : %p\n", configuration.device);
  return 4;
}
