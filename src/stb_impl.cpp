// stb_impl.cpp -- the ONE translation unit that holds the stb implementations.
// Every other file only includes the stb headers for declarations.
// Authors: <TODO: names / student IDs>
// External sources: stb_image.h / stb_image_write.h (public domain, Sean
//                   Barrett, https://github.com/nothings/stb).
// Third-party code: silence its warnings so ours stay visible.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
