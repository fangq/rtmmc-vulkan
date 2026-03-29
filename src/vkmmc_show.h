/*
 * vkmmc_show.h — Interactive Vulkan viewer for rtmmc-vulkan
 * Included only when USE_SHOW is defined (build with: make SHOW=1)
 * Activated at runtime with --show flag.
 *
 * Provides a GLFW/Vulkan window that:
 *   1. Renders the simulation mesh domain (semi-transparent triangles)
 *   2. Shows a "Start Simulation" button in NDC space
 *   3. When button or SPACE is pressed, runs the simulation callback
 *   4. Renders trajectory lines over the mesh after simulation completes
 */

#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <array>
#include <functional>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "vkmmc_show_vert_spv.h"
#include "vkmmc_show_frag_spv.h"

/* ================================================================ */
/*                          Data structs                            */
/* ================================================================ */

struct ShowVert {
    float x, y, z, nx, ny, nz;
};

struct ShowTraj {
    std::vector<float>    pts;   // xyz per event (size = N*3)
    std::vector<uint32_t> ids;   // photon ID per event (size = N)
    std::vector<uint32_t> evts;  // event type: 0=launch 1=scatter 2=term 3=boundary
};

struct ShowMesh {
    std::vector<ShowVert>   verts;
    std::vector<uint32_t>   inds;
    float cx, cy, cz, radius;
    std::vector<float>      src_lines; /* LINE_LIST: flat xyz triples, no normals */
};

/* Push constant: mat4 (64 bytes) + vec4 (16 bytes) = 80 bytes */
struct ShowPC {
    float mvp[16];
    float color[4];
};

/* ================================================================ */
/*                         Math helpers                             */
/* ================================================================ */

static void m4_id(float* m) {
    memset(m, 0, 64);
    m[0] = m[5] = m[10] = m[15] = 1.f;
}

/* Column-major 4x4 multiply: r = a * b */
static void m4_mul(float* r, const float* a, const float* b) {
    float tmp[16];

    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float s = 0.f;

            for (int k = 0; k < 4; k++) {
                s += a[k * 4 + row] * b[col * 4 + k];
            }

            tmp[col * 4 + row] = s;
        }

    memcpy(r, tmp, 64);
}

/* Perspective projection with Y-flip for Vulkan NDC */
static void m4_persp(float* m, float fovY, float asp, float nr, float fr) {
    memset(m, 0, 64);
    float f = 1.f / tanf(fovY * 0.5f);
    m[0]  =  f / asp;
    m[5]  = -f;          /* Y-flip for Vulkan */
    m[10] = fr / (nr - fr);
    m[11] = -1.f;
    m[14] = (fr * nr) / (nr - fr);
}

/* LookAt: eye=(ex,ey,ez), center=(cx,cy,cz), up=(0,1,0) */
static void m4_lookat(float* m, float ex, float ey, float ez,
                      float cx, float cy, float cz) {
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float fl = sqrtf(fx * fx + fy * fy + fz * fz);

    if (fl < 1e-10f) {
        fl = 1e-10f;
    }

    fx /= fl;
    fy /= fl;
    fz /= fl;

    /* right = forward x up(0,1,0) */
    float rx = fz, ry = 0.f, rz = -fx;
    float rl = sqrtf(rx * rx + ry * ry + rz * rz);

    if (rl < 1e-10f) {
        rx = 1.f;
        ry = rz = 0.f;
    } else {
        rx /= rl;
        ry /= rl;
        rz /= rl;
    }

    /* true up = right x forward */
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    memset(m, 0, 64);
    /* column-major */
    m[0] = rx;
    m[1] = ux;
    m[2] = -fx;
    m[3] = 0.f;
    m[4] = ry;
    m[5] = uy;
    m[6] = -fy;
    m[7] = 0.f;
    m[8] = rz;
    m[9] = uz;
    m[10] = -fz;
    m[11] = 0.f;
    m[12] = -(rx * ex + ry * ey + rz * ez);
    m[13] = -(ux * ex + uy * ey + uz * ez);
    m[14] =  (fx * ex + fy * ey + fz * ez);
    m[15] = 1.f;
}

/* ================================================================ */
/*                           Camera                                 */
/* ================================================================ */

struct Camera {
    float cx = 0, cy = 0, cz = 0;
    float dist = 100.f;
    float yaw = 0.3f, pitch = 0.3f;
    int W = 1280, H = 720;

    void mvp(float* out) const {
        float eye[3] = {
            cx + dist * cosf(pitch)* sinf(yaw),
            cy + dist * sinf(pitch),
            cz + dist * cosf(pitch)* cosf(yaw)
        };

        float proj[16], view[16];
        m4_persp(proj, 0.7854f /* 45 deg */, (float)W / (float)H, dist * 0.001f, dist * 10.f);
        m4_lookat(view, eye[0], eye[1], eye[2], cx, cy, cz);
        m4_mul(out, proj, view);
    }
};

/* ================================================================ */
/*                       Vulkan buffer helper                       */
/* ================================================================ */

struct ShowBuf {
    VkDevice        dev = VK_NULL_HANDLE;
    VkBuffer        buf = VK_NULL_HANDLE;
    VkDeviceMemory  mem = VK_NULL_HANDLE;
    VkDeviceSize    sz  = 0;

    void destroy() {
        if (buf) {
            vkDestroyBuffer(dev, buf, nullptr);
            buf = VK_NULL_HANDLE;
        }

        if (mem) {
            vkFreeMemory(dev, mem, nullptr);
            mem = VK_NULL_HANDLE;
        }

        sz = 0;
    }
};

/* ================================================================ */
/*                         ShowCtx                                  */
/* ================================================================ */

struct ShowCtx {
    VkInstance       inst      = VK_NULL_HANDLE;
    VkSurfaceKHR     surf      = VK_NULL_HANDLE;
    VkPhysicalDevice pdev      = VK_NULL_HANDLE;
    VkDevice         dev       = VK_NULL_HANDLE;
    uint32_t         gqf = 0, pqf = 0;
    VkQueue          gq  = VK_NULL_HANDLE, pq = VK_NULL_HANDLE;

    VkSwapchainKHR        swap    = VK_NULL_HANDLE;
    VkFormat              swapFmt = VK_FORMAT_UNDEFINED;
    VkExtent2D            ext     = {0, 0};
    std::vector<VkImage>     imgs;
    std::vector<VkImageView> views;

    VkImage        depth    = VK_NULL_HANDLE;
    VkDeviceMemory depthMem = VK_NULL_HANDLE;
    VkImageView    depthView = VK_NULL_HANDLE;

    VkRenderPass               rp   = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> fbs;

    VkShaderModule   vsm = VK_NULL_HANDLE, fsm = VK_NULL_HANDLE;
    VkPipelineLayout playout  = VK_NULL_HANDLE;
    VkPipeline       triPipe  = VK_NULL_HANDLE;
    VkPipeline       linePipe = VK_NULL_HANDLE;

    VkCommandPool   pool = VK_NULL_HANDLE;
    static const int FRAMES = 2;
    VkCommandBuffer cmd[2]    = {};
    VkSemaphore     imgSem[2] = {}, renSem[2] = {};
    VkFence         fence[2]  = {};

    ShowBuf meshVB, meshIB, trajVB, btnVB, srcVB;
    uint32_t meshVtxCnt = 0, meshIdxCnt = 0, trajPtCnt = 0, srcPtCnt = 0;
};

/* ================================================================ */
/*                       VK helper macro                            */
/* ================================================================ */

#define SH_VK_CHECK(x) do { \
        VkResult _r = (x); \
        if (_r != VK_SUCCESS) { \
            fprintf(stderr, "[show] Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
            abort(); \
        } \
    } while(0)

/* ================================================================ */
/*                    Vulkan helper functions                        */
/* ================================================================ */

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }

    fprintf(stderr, "[show] No suitable memory type\n");
    abort();
}

static ShowBuf make_buf(ShowCtx& sc, VkDeviceSize sz, VkBufferUsageFlags usage,
                        const void* data = nullptr) {
    ShowBuf sb;
    sb.dev = sc.dev;
    sb.sz  = sz;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = sz;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    SH_VK_CHECK(vkCreateBuffer(sc.dev, &bci, nullptr, &sb.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(sc.dev, sb.buf, &req);

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_mem(sc.pdev, req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    SH_VK_CHECK(vkAllocateMemory(sc.dev, &mai, nullptr, &sb.mem));
    SH_VK_CHECK(vkBindBufferMemory(sc.dev, sb.buf, sb.mem, 0));

    if (data) {
        void* p;
        SH_VK_CHECK(vkMapMemory(sc.dev, sb.mem, 0, sz, 0, &p));
        memcpy(p, data, (size_t)sz);
        vkUnmapMemory(sc.dev, sb.mem);
    }

    return sb;
}

static VkFormat find_depth_format(VkPhysicalDevice pd) {
    VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM
    };

    for (VkFormat fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(pd, fmt, &props);

        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return fmt;
        }
    }

    fprintf(stderr, "[show] No suitable depth format\n");
    abort();
}

static void create_img(ShowCtx& sc, uint32_t w, uint32_t h, VkFormat fmt,
                       VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = fmt;
    ici.extent      = {w, h, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    SH_VK_CHECK(vkCreateImage(sc.dev, &ici, nullptr, &sc.depth));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(sc.dev, sc.depth, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_mem(sc.pdev, req.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    SH_VK_CHECK(vkAllocateMemory(sc.dev, &mai, nullptr, &sc.depthMem));
    SH_VK_CHECK(vkBindImageMemory(sc.dev, sc.depth, sc.depthMem, 0));

    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = sc.depth;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = fmt;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    SH_VK_CHECK(vkCreateImageView(sc.dev, &vci, nullptr, &sc.depthView));
}

static void create_swapchain(ShowCtx& sc, GLFWwindow* win) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(sc.pdev, sc.surf, &caps);

    /* Choose format */
    uint32_t fmtCnt = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(sc.pdev, sc.surf, &fmtCnt, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCnt);
    vkGetPhysicalDeviceSurfaceFormatsKHR(sc.pdev, sc.surf, &fmtCnt, fmts.data());

    sc.swapFmt = fmts[0].format;
    VkColorSpaceKHR colorSpace = fmts[0].colorSpace;

    for (auto& f : fmts) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_B8G8R8A8_UNORM)
                && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            sc.swapFmt = f.format;
            colorSpace = f.colorSpace;
            break;
        }
    }

    /* Choose extent */
    if (caps.currentExtent.width != UINT32_MAX) {
        sc.ext = caps.currentExtent;
    } else {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        sc.ext.width  = (uint32_t)std::max(caps.minImageExtent.width,
                                           std::min(caps.maxImageExtent.width, (uint32_t)w));
        sc.ext.height = (uint32_t)std::max(caps.minImageExtent.height,
                                           std::min(caps.maxImageExtent.height, (uint32_t)h));
    }

    uint32_t imgCount = caps.minImageCount + 1;

    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount) {
        imgCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface          = sc.surf;
    sci.minImageCount    = imgCount;
    sci.imageFormat      = sc.swapFmt;
    sci.imageColorSpace  = colorSpace;
    sci.imageExtent      = sc.ext;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;

    if (sc.gqf != sc.pqf) {
        uint32_t qfi[] = {sc.gqf, sc.pqf};
        sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices   = qfi;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    SH_VK_CHECK(vkCreateSwapchainKHR(sc.dev, &sci, nullptr, &sc.swap));

    /* Get images */
    uint32_t cnt = 0;
    vkGetSwapchainImagesKHR(sc.dev, sc.swap, &cnt, nullptr);
    sc.imgs.resize(cnt);
    vkGetSwapchainImagesKHR(sc.dev, sc.swap, &cnt, sc.imgs.data());

    /* Create image views */
    sc.views.resize(cnt);

    for (uint32_t i = 0; i < cnt; i++) {
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image    = sc.imgs[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = sc.swapFmt;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        SH_VK_CHECK(vkCreateImageView(sc.dev, &vci, nullptr, &sc.views[i]));
    }
}

static void create_rp(ShowCtx& sc) {
    VkFormat depthFmt = find_depth_format(sc.pdev);

    VkAttachmentDescription att[2] = {};
    /* Color */
    att[0].format         = sc.swapFmt;
    att[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    /* Depth */
    att[1].format         = depthFmt;
    att[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub = {};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep = {};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rci.attachmentCount = 2;
    rci.pAttachments    = att;
    rci.subpassCount    = 1;
    rci.pSubpasses      = &sub;
    rci.dependencyCount = 1;
    rci.pDependencies   = &dep;
    SH_VK_CHECK(vkCreateRenderPass(sc.dev, &rci, nullptr, &sc.rp));
}

static void create_fbs(ShowCtx& sc) {
    sc.fbs.resize(sc.views.size());

    for (size_t i = 0; i < sc.views.size(); i++) {
        VkImageView attachments[] = {sc.views[i], sc.depthView};
        VkFramebufferCreateInfo fci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass      = sc.rp;
        fci.attachmentCount = 2;
        fci.pAttachments    = attachments;
        fci.width           = sc.ext.width;
        fci.height          = sc.ext.height;
        fci.layers          = 1;
        SH_VK_CHECK(vkCreateFramebuffer(sc.dev, &fci, nullptr, &sc.fbs[i]));
    }
}

static VkShaderModule make_shader(VkDevice dev, const uint8_t* spv, size_t sz) {
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sz;
    smi.pCode    = reinterpret_cast<const uint32_t*>(spv);
    VkShaderModule mod;
    SH_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
    return mod;
}

static VkPipeline create_pipe(ShowCtx& sc, VkPrimitiveTopology topo, bool blend,
                              bool has_normal = false) {
    VkVertexInputBindingDescription vbind = {};
    vbind.binding   = 0;
    vbind.stride    = has_normal ? 24u : 12u; /* 6 or 3 floats */
    vbind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vattrs[2] = {};
    vattrs[0].location = 0;
    vattrs[0].binding  = 0;
    vattrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    vattrs[0].offset   = 0;
    vattrs[1].location = 1;
    vattrs[1].binding  = 0;
    vattrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    vattrs[1].offset   = 12; /* after xyz */

    VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vis.vertexBindingDescriptionCount   = 1;
    vis.pVertexBindingDescriptions      = &vbind;
    vis.vertexAttributeDescriptionCount = has_normal ? 2u : 1u;
    vis.pVertexAttributeDescriptions    = vattrs;

    VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ias.topology = topo;

    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo ras = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    ras.polygonMode = VK_POLYGON_MODE_FILL;
    ras.cullMode    = VK_CULL_MODE_NONE;
    ras.frontFace   = VK_FRONT_FACE_CLOCKWISE;
    ras.lineWidth   = 1.0f;  /* >1.0 requires wideLines feature; 1.0 is always valid */

    VkPipelineMultisampleStateCreateInfo mss = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    mss.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable  = VK_TRUE;
    dss.depthWriteEnable = VK_TRUE;
    dss.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba = {};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    if (blend) {
        cba.blendEnable         = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp        = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1;
    cbs.pAttachments    = &cba;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = sc.vsm;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = sc.fsm;
    stages[1].pName  = "main";

    VkGraphicsPipelineCreateInfo pci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vis;
    pci.pInputAssemblyState = &ias;
    pci.pViewportState      = &vps;
    pci.pRasterizationState = &ras;
    pci.pMultisampleState   = &mss;
    pci.pDepthStencilState  = &dss;
    pci.pColorBlendState    = &cbs;
    pci.pDynamicState       = &dyn;
    pci.layout              = sc.playout;
    pci.renderPass          = sc.rp;
    pci.subpass             = 0;

    VkPipeline pipe;
    SH_VK_CHECK(vkCreateGraphicsPipelines(sc.dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe));
    return pipe;
}

static void init_viewer(ShowCtx& sc, GLFWwindow* win, uint32_t w, uint32_t h) {
    /* Instance */
    uint32_t glfwExtCnt = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCnt);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rtmmc-vulkan-show";
    app.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo        = &app;
    ici.enabledExtensionCount   = glfwExtCnt;
    ici.ppEnabledExtensionNames = glfwExts;
    SH_VK_CHECK(vkCreateInstance(&ici, nullptr, &sc.inst));

    /* Surface */
    SH_VK_CHECK(glfwCreateWindowSurface(sc.inst, win, nullptr, &sc.surf));

    /* Physical device: prefer discrete GPU that supports graphics + present */
    uint32_t pdCnt = 0;
    vkEnumeratePhysicalDevices(sc.inst, &pdCnt, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCnt);
    vkEnumeratePhysicalDevices(sc.inst, &pdCnt, pds.data());

    sc.pdev   = VK_NULL_HANDLE;
    sc.gqf    = UINT32_MAX;
    sc.pqf    = UINT32_MAX;

    auto pick_queues = [&](VkPhysicalDevice pd, uint32_t& gqf_out, uint32_t& pqf_out) -> bool {
        uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfc);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, qfs.data());
        uint32_t gq = UINT32_MAX, pq = UINT32_MAX;

        for (uint32_t i = 0; i < qfc; i++) {
            if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                gq = i;
            }

            VkBool32 ps = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, sc.surf, &ps);

            if (ps) {
                pq = i;
            }

            if (gq != UINT32_MAX && pq != UINT32_MAX) {
                break;
            }
        }

        if (gq == UINT32_MAX || pq == UINT32_MAX) {
            return false;
        }

        gqf_out = gq;
        pqf_out = pq;
        return true;
    };

    /* Try discrete first */
    for (auto& pd : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(pd, &pr);
        uint32_t gq, pq;

        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pick_queues(pd, gq, pq)) {
            sc.pdev = pd;
            sc.gqf = gq;
            sc.pqf = pq;
            break;
        }
    }

    /* Fallback to any */
    if (sc.pdev == VK_NULL_HANDLE) {
        for (auto& pd : pds) {
            uint32_t gq, pq;

            if (pick_queues(pd, gq, pq)) {
                sc.pdev = pd;
                sc.gqf = gq;
                sc.pqf = pq;
                break;
            }
        }
    }

    if (sc.pdev == VK_NULL_HANDLE) {
        fprintf(stderr, "[show] No suitable GPU for viewer\n");
        abort();
    }

    /* Logical device */
    float prio = 1.f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    {
        VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = sc.gqf;
        q.queueCount = 1;
        q.pQueuePriorities = &prio;
        qcis.push_back(q);

        if (sc.pqf != sc.gqf) {
            q.queueFamilyIndex = sc.pqf;
            qcis.push_back(q);
        }
    }
    const char* devExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    /* Enable bufferDeviceAddress on the graphics device to match the compute context.
     * On NVIDIA, once any logical device on a physical GPU allocates a buffer with
     * VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT the driver switches the whole GPU into
     * a different VA/page-table mode.  A co-existing logical device that was created
     * WITHOUT bufferDeviceAddress becomes incompatible and returns VK_ERROR_DEVICE_LOST
     * on its next vkQueueSubmit.  Enabling the feature here keeps both devices in the
     * same mode from the beginning. */
    VkPhysicalDeviceVulkan12Features v12f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12f.bufferDeviceAddress = VK_TRUE;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount    = (uint32_t)qcis.size();
    dci.pQueueCreateInfos       = qcis.data();
    dci.enabledExtensionCount   = 1;
    dci.ppEnabledExtensionNames = devExts;
    dci.pNext                   = &v12f;
    SH_VK_CHECK(vkCreateDevice(sc.pdev, &dci, nullptr, &sc.dev));

    vkGetDeviceQueue(sc.dev, sc.gqf, 0, &sc.gq);
    vkGetDeviceQueue(sc.dev, sc.pqf, 0, &sc.pq);

    /* Swapchain + depth + render pass + framebuffers */
    create_swapchain(sc, win);
    VkFormat depthFmt = find_depth_format(sc.pdev);
    create_img(sc, sc.ext.width, sc.ext.height, depthFmt,
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               VK_IMAGE_ASPECT_DEPTH_BIT);
    create_rp(sc);
    create_fbs(sc);

    /* Shaders */
    sc.vsm = make_shader(sc.dev, vkmmc_show_vert_spv, vkmmc_show_vert_spv_len);
    sc.fsm = make_shader(sc.dev, vkmmc_show_frag_spv, vkmmc_show_frag_spv_len);

    /* Pipeline layout with 80-byte push constant */
    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset     = 0;
    pcr.size       = 80;

    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    SH_VK_CHECK(vkCreatePipelineLayout(sc.dev, &plci, nullptr, &sc.playout));

    /* Pipelines: triPipe uses stride-24 (xyz+normal); linePipe uses stride-12 (xyz only) */
    sc.triPipe  = create_pipe(sc, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true,  true);
    sc.linePipe = create_pipe(sc, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,     false, false);

    /* Command pool + command buffers */
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = sc.gqf;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    SH_VK_CHECK(vkCreateCommandPool(sc.dev, &cpi, nullptr, &sc.pool));

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = sc.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = ShowCtx::FRAMES;
    SH_VK_CHECK(vkAllocateCommandBuffers(sc.dev, &cai, sc.cmd));

    /* Sync objects */
    VkSemaphoreCreateInfo semi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fenci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < ShowCtx::FRAMES; i++) {
        SH_VK_CHECK(vkCreateSemaphore(sc.dev, &semi, nullptr, &sc.imgSem[i]));
        SH_VK_CHECK(vkCreateSemaphore(sc.dev, &semi, nullptr, &sc.renSem[i]));
        SH_VK_CHECK(vkCreateFence(sc.dev, &fenci, nullptr, &sc.fence[i]));
    }

    printf("[show] Viewer initialized (%ux%u)\n", w, h);
}

/* ================================================================ */
/*                      Upload helpers                              */
/* ================================================================ */

static void upload_mesh(ShowCtx& sc, const ShowMesh& mesh) {
    if (mesh.verts.empty()) {
        return;
    }

    sc.meshVB.destroy();
    sc.meshIB.destroy();
    sc.meshVB = make_buf(sc, mesh.verts.size() * sizeof(ShowVert),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.verts.data());
    sc.meshVtxCnt = (uint32_t)mesh.verts.size();

    if (!mesh.inds.empty()) {
        sc.meshIB = make_buf(sc, mesh.inds.size() * sizeof(uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.inds.data());
        sc.meshIdxCnt = (uint32_t)mesh.inds.size();
    }
}

static void upload_traj(ShowCtx& sc, const ShowTraj& traj) {
    sc.trajVB.destroy();
    sc.trajPtCnt = 0;

    if (traj.pts.empty() || traj.ids.empty()) {
        return;
    }

    size_t N = traj.ids.size();
    bool have_evts = (traj.evts.size() == N);

    /* Sort event indices by photon ID, preserving intra-photon order.
     * Mirrors mcxplotphotons: [newid, idx] = sort(traj.id)
     * GPU atomicAdd assigns slots in random order across photons, so without
     * this sort almost no consecutive pairs share the same photon ID. */
    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
    [&](uint32_t a, uint32_t b) {
        return traj.ids[a] < traj.ids[b];
    });

    /* Pass 1: mark photons that ever experienced Fresnel reflection (event 4). */
    std::vector<bool> reflected;

    if (have_evts) {
        uint32_t max_id = *std::max_element(traj.ids.begin(), traj.ids.end());
        reflected.assign(max_id + 1, false);

        for (size_t i = 0; i < N; i++)
            if (traj.evts[i] == 4u) {
                reflected[traj.ids[i]] = true;
            }
    }

    /* Pass 2: build LINE_LIST segments using sorted order. */
    std::vector<float> linePts;
    std::vector<float> launchPts;
    linePts.reserve(N * 6);

    /* Size of launch cross arms — 2% of trajectory z-span */
    float z0 = traj.pts[2], z1 = traj.pts[2];

    for (size_t i = 0; i < N; i++) {
        float z = traj.pts[i * 3 + 2];

        if (z < z0) {
            z0 = z;
        }

        if (z > z1) {
            z1 = z;
        }
    }

    float cs = (z1 - z0) * 0.02f;

    if (cs < 0.5f) {
        cs = 0.5f;
    }

    for (size_t j = 0; j + 1 < N; j++) {
        uint32_t i0 = order[j], i1 = order[j + 1];

        if (traj.ids[i0] != traj.ids[i1]) {
            continue;    /* different photons */
        }

        uint32_t pid = traj.ids[i0];

        if (have_evts && pid < (uint32_t)reflected.size() && reflected[pid]) {
            continue;
        }

        linePts.push_back(traj.pts[i0 * 3 + 0]);
        linePts.push_back(traj.pts[i0 * 3 + 1]);
        linePts.push_back(traj.pts[i0 * 3 + 2]);
        linePts.push_back(traj.pts[i1 * 3 + 0]);
        linePts.push_back(traj.pts[i1 * 3 + 1]);
        linePts.push_back(traj.pts[i1 * 3 + 2]);
    }

    /* Collect launch markers (event type 0) for non-reflected photons */
    if (have_evts) {
        for (size_t j = 0; j < N; j++) {
            uint32_t i = order[j];

            if (traj.evts[i] != 0) {
                continue;
            }

            uint32_t pid = traj.ids[i];

            if (pid < (uint32_t)reflected.size() && reflected[pid]) {
                continue;
            }

            float lx = traj.pts[i * 3 + 0];
            float ly = traj.pts[i * 3 + 1];
            float lz = traj.pts[i * 3 + 2];
            launchPts.push_back(lx - cs);
            launchPts.push_back(ly);
            launchPts.push_back(lz);
            launchPts.push_back(lx + cs);
            launchPts.push_back(ly);
            launchPts.push_back(lz);
            launchPts.push_back(lx);
            launchPts.push_back(ly - cs);
            launchPts.push_back(lz);
            launchPts.push_back(lx);
            launchPts.push_back(ly + cs);
            launchPts.push_back(lz);
            launchPts.push_back(lx);
            launchPts.push_back(ly);
            launchPts.push_back(lz - cs);
            launchPts.push_back(lx);
            launchPts.push_back(ly);
            launchPts.push_back(lz + cs);
        }
    }

    for (float v : launchPts) {
        linePts.push_back(v);
    }

    if (linePts.empty()) {
        return;
    }

    sc.trajVB = make_buf(sc, linePts.size() * sizeof(float),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, linePts.data());
    sc.trajPtCnt = (uint32_t)(linePts.size() / 3);
}

static void upload_src(ShowCtx& sc, const std::vector<float>& lines) {
    sc.srcVB.destroy();
    sc.srcPtCnt = 0;

    if (lines.empty()) {
        return;
    }

    sc.srcVB = make_buf(sc, lines.size() * sizeof(float),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, lines.data());
    sc.srcPtCnt = (uint32_t)(lines.size() / 3);
}

static void upload_button(ShowCtx& sc, int state) {
    (void)state;
    sc.btnVB.destroy();

    /* Button quad in NDC: x=[0.62,0.97].
     * With the Y-flip viewport (vp.y=H, vp.height=-H), NDC y=-1 → screen bottom.
     * Use y∈[-0.97,-0.72] so the button appears at the bottom-right of the window.
     * Hover detection in cb_cursor uses screen-space NDC [0.72,0.97] which maps
     * to the same screen region. */
    float x0 = 0.62f, x1 = 0.97f, y0 = -0.97f, y1 = -0.72f, z = 0.5f;
    /* Play triangle (▶) inside button */
    float px0 = 0.71f, px1 = 0.88f, pym = (y0 + y1) * 0.5f;
    float pyt = y0 + (y1 - y0) * 0.2f, pyb = y1 - (y1 - y0) * 0.2f;
    float pz  = 0.49f; /* slightly in front (smaller z) of button quad */

    /* Each vertex: xyz + normal(0,0,1) — forward-facing for unlit look */
#define BTN_V(vx,vy,vz) (vx),(vy),(vz), 0.f,0.f,1.f
    float verts[] = {
        /* Quad: 2 triangles CW */
        BTN_V(x0, y0, z),
        BTN_V(x1, y0, z),
        BTN_V(x0, y1, z),
        BTN_V(x1, y0, z),
        BTN_V(x1, y1, z),
        BTN_V(x0, y1, z),
        /* Play triangle CW */
        BTN_V(px0, pyt, pz),
        BTN_V(px1, pym, pz),
        BTN_V(px0, pyb, pz)
    };
#undef BTN_V

    sc.btnVB = make_buf(sc, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, verts);
}

/* ================================================================ */
/*                        draw_frame                                */
/* ================================================================ */

static uint32_t g_frame_count = 0;

/* Returns false if the device is lost (e.g. GPU watchdog fired during simulation). */
static bool draw_frame(ShowCtx& sc, const Camera& cam, int state, bool has_traj) {
    int frame = (int)(g_frame_count % (uint32_t)ShowCtx::FRAMES);

    vkWaitForFences(sc.dev, 1, &sc.fence[frame], VK_TRUE, UINT64_MAX);

    uint32_t imgIdx = 0;
    VkResult acq = vkAcquireNextImageKHR(sc.dev, sc.swap, UINT64_MAX,
                                         sc.imgSem[frame], VK_NULL_HANDLE, &imgIdx);

    if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
        return true;
    }

    if (acq != VK_SUCCESS) {
        return true;
    }

    vkResetFences(sc.dev, 1, &sc.fence[frame]);

    VkCommandBuffer cmd = sc.cmd[frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    SH_VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    VkClearValue clears[2];
    clears[0].color.float32[0] = 0.1f;
    clears[0].color.float32[1] = 0.1f;
    clears[0].color.float32[2] = 0.15f;
    clears[0].color.float32[3] = 1.0f;
    clears[1].depthStencil.depth   = 1.0f;
    clears[1].depthStencil.stencil = 0;

    VkRenderPassBeginInfo rpbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass      = sc.rp;
    rpbi.framebuffer     = sc.fbs[imgIdx];
    rpbi.renderArea      = {{0, 0}, sc.ext};
    rpbi.clearValueCount = 2;
    rpbi.pClearValues    = clears;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    /* Dynamic viewport with Y-flip for Vulkan NDC convention */
    VkViewport vp = {};
    vp.x        = 0.f;
    vp.y        = (float)sc.ext.height;
    vp.width    = (float)sc.ext.width;
    vp.height   = -(float)sc.ext.height;
    vp.minDepth = 0.f;
    vp.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor = {{0, 0}, sc.ext};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    /* --- Draw mesh (semi-transparent triangles) --- */
    if (sc.meshIdxCnt > 0 && sc.meshVB.buf) {
        ShowPC pc;
        cam.mvp(pc.mvp);
        pc.color[0] = 0.5f;
        pc.color[1] = 0.7f;
        pc.color[2] = 1.0f;
        pc.color[3] = 0.25f;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sc.triPipe);
        vkCmdPushConstants(cmd, sc.playout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 80, &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sc.meshVB.buf, &off);
        vkCmdBindIndexBuffer(cmd, sc.meshIB.buf, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, sc.meshIdxCnt, 1, 0, 0, 0);
    }

    /* --- Draw source indicator (always visible) --- */
    if (sc.srcPtCnt > 0 && sc.srcVB.buf) {
        ShowPC pc;
        cam.mvp(pc.mvp);
        /* Bright yellow for source position/direction */
        pc.color[0] = 1.0f;
        pc.color[1] = 0.9f;
        pc.color[2] = 0.0f;
        pc.color[3] = 1.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sc.linePipe);
        vkCmdPushConstants(cmd, sc.playout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 80, &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sc.srcVB.buf, &off);
        vkCmdDraw(cmd, sc.srcPtCnt, 1, 0, 0);
    }

    /* --- Draw trajectory lines --- */
    if (has_traj && sc.trajPtCnt > 0 && sc.trajVB.buf) {
        ShowPC pc;
        cam.mvp(pc.mvp);
        pc.color[0] = 1.0f;
        pc.color[1] = 0.6f;
        pc.color[2] = 0.1f;
        pc.color[3] = 1.0f;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sc.linePipe);
        vkCmdPushConstants(cmd, sc.playout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 80, &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sc.trajVB.buf, &off);
        vkCmdDraw(cmd, sc.trajPtCnt, 1, 0, 0);
    }

    /* --- Draw button in NDC space (identity MVP) --- */
    if (sc.btnVB.buf) {
        ShowPC pc;
        m4_id(pc.mvp);

        /* State colors for background quad */
        if (state == 0) {
            pc.color[0] = 0.2f;
            pc.color[1] = 0.75f;
            pc.color[2] = 0.2f;
            pc.color[3] = 0.85f;
        } else if (state == 1) {
            pc.color[0] = 0.8f;
            pc.color[1] = 0.7f;
            pc.color[2] = 0.1f;
            pc.color[3] = 0.85f;
        } else {
            pc.color[0] = 0.2f;
            pc.color[1] = 0.5f;
            pc.color[2] = 0.9f;
            pc.color[3] = 0.85f;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sc.triPipe);
        vkCmdPushConstants(cmd, sc.playout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 80, &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sc.btnVB.buf, &off);
        vkCmdDraw(cmd, 6, 1, 0, 0); /* quad: 6 verts */

        /* Play icon — darker shade */
        if (state == 0) {
            pc.color[0] = 0.05f;
            pc.color[1] = 0.4f;
            pc.color[2] = 0.05f;
            pc.color[3] = 1.0f;
        } else if (state == 1) {
            pc.color[0] = 0.5f;
            pc.color[1] = 0.4f;
            pc.color[2] = 0.02f;
            pc.color[3] = 1.0f;
        } else {
            pc.color[0] = 0.05f;
            pc.color[1] = 0.2f;
            pc.color[2] = 0.6f;
            pc.color[3] = 1.0f;
        }

        vkCmdPushConstants(cmd, sc.playout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 80, &pc);
        vkCmdDraw(cmd, 3, 1, 6, 0); /* play triangle: 3 verts starting at vertex 6 */
    }

    vkCmdEndRenderPass(cmd);
    SH_VK_CHECK(vkEndCommandBuffer(cmd));

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &sc.imgSem[frame];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &sc.renSem[frame];
    VkResult sr = vkQueueSubmit(sc.gq, 1, &si, sc.fence[frame]);

    if (sr == VK_ERROR_DEVICE_LOST) {
        fprintf(stderr, "[show] GPU device lost after simulation (watchdog timeout?).\n"
                "       Use a smaller -B batch size (e.g. -B 100000) to stay\n"
                "       within the GPU watchdog window.\n");
        return false;
    }

    SH_VK_CHECK(sr);

    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &sc.renSem[frame];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &sc.swap;
    pi.pImageIndices      = &imgIdx;
    VkResult pr = vkQueuePresentKHR(sc.pq, &pi);

    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        /* will be handled on next frame */
    }

    return true;
}

/* ================================================================ */
/*                    GLFW input state + callbacks                  */
/* ================================================================ */

struct InputState {
    bool   leftDown = false, rightDown = false;
    double lastX = 0, lastY = 0;
    float  dYaw = 0, dPitch = 0, dScroll = 0;
    bool   startSim = false, btnHovered = false;
    double mouseX = 0, mouseY = 0;
    bool   clicked = false;
    int    W = 1280, H = 720;
    bool   resized = false;
};

static void cb_cursor(GLFWwindow* win, double x, double y) {
    InputState* inp = (InputState*)glfwGetWindowUserPointer(win);
    inp->mouseX = x;
    inp->mouseY = y;

    if (inp->leftDown) {
        double dx = x - inp->lastX, dy = y - inp->lastY;
        inp->dYaw   += (float)(dx * 0.005);
        inp->dPitch += (float)(dy * 0.005);
    }

    inp->lastX = x;
    inp->lastY = y;

    /* Check button hover: NDC x=[0.62,0.97], y=[0.72,0.97] */
    double nx = (x / inp->W) * 2.0 - 1.0;
    double ny = (y / inp->H) * 2.0 - 1.0;
    inp->btnHovered = (nx >= 0.62 && nx <= 0.97 && ny >= 0.72 && ny <= 0.97);
}

static void cb_mouse_btn(GLFWwindow* win, int btn, int action, int /*mods*/) {
    InputState* inp = (InputState*)glfwGetWindowUserPointer(win);

    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        inp->leftDown = (action == GLFW_PRESS);

        if (action == GLFW_PRESS && inp->btnHovered) {
            inp->clicked = true;
        }
    }

    if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
        inp->rightDown = (action == GLFW_PRESS);
    }
}

static void cb_scroll(GLFWwindow* win, double /*dx*/, double dy) {
    InputState* inp = (InputState*)glfwGetWindowUserPointer(win);
    inp->dScroll += (float)dy;
}

static void cb_key(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
    InputState* inp = (InputState*)glfwGetWindowUserPointer(win);

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_SPACE) {
            inp->startSim = true;
        }

        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }
    }
}

static void cb_resize(GLFWwindow* win, int w, int h) {
    InputState* inp = (InputState*)glfwGetWindowUserPointer(win);
    inp->W = w;
    inp->H = h;
    inp->resized = true;
}

/* ================================================================ */
/*                    Swapchain recreation                          */
/* ================================================================ */

static void destroy_swap_resources(ShowCtx& sc) {
    for (auto fb : sc.fbs) {
        vkDestroyFramebuffer(sc.dev, fb, nullptr);
    }

    sc.fbs.clear();

    if (sc.rp) {
        vkDestroyRenderPass(sc.dev, sc.rp, nullptr);
        sc.rp = VK_NULL_HANDLE;
    }

    if (sc.depthView) {
        vkDestroyImageView(sc.dev, sc.depthView, nullptr);
        sc.depthView = VK_NULL_HANDLE;
    }

    if (sc.depth)    {
        vkDestroyImage(sc.dev, sc.depth, nullptr);
        sc.depth = VK_NULL_HANDLE;
    }

    if (sc.depthMem) {
        vkFreeMemory(sc.dev, sc.depthMem, nullptr);
        sc.depthMem = VK_NULL_HANDLE;
    }

    for (auto v : sc.views) {
        vkDestroyImageView(sc.dev, v, nullptr);
    }

    sc.views.clear();

    if (sc.swap) {
        vkDestroySwapchainKHR(sc.dev, sc.swap, nullptr);
        sc.swap = VK_NULL_HANDLE;
    }
}

static void recreate_swap(ShowCtx& sc, GLFWwindow* win) {
    vkDeviceWaitIdle(sc.dev);
    destroy_swap_resources(sc);
    create_swapchain(sc, win);
    VkFormat depthFmt = find_depth_format(sc.pdev);
    create_img(sc, sc.ext.width, sc.ext.height, depthFmt,
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               VK_IMAGE_ASPECT_DEPTH_BIT);
    create_rp(sc);
    create_fbs(sc);
}

/* ================================================================ */
/*                         Cleanup                                  */
/* ================================================================ */

static void destroy_viewer(ShowCtx& sc) {
    vkDeviceWaitIdle(sc.dev);

    sc.meshVB.destroy();
    sc.meshIB.destroy();
    sc.trajVB.destroy();
    sc.btnVB.destroy();
    sc.srcVB.destroy();

    for (int i = 0; i < ShowCtx::FRAMES; i++) {
        vkDestroyFence(sc.dev, sc.fence[i], nullptr);
        vkDestroySemaphore(sc.dev, sc.renSem[i], nullptr);
        vkDestroySemaphore(sc.dev, sc.imgSem[i], nullptr);
    }

    if (sc.pool) {
        vkDestroyCommandPool(sc.dev, sc.pool, nullptr);
    }

    if (sc.linePipe) {
        vkDestroyPipeline(sc.dev, sc.linePipe, nullptr);
    }

    if (sc.triPipe) {
        vkDestroyPipeline(sc.dev, sc.triPipe, nullptr);
    }

    if (sc.playout) {
        vkDestroyPipelineLayout(sc.dev, sc.playout, nullptr);
    }

    if (sc.fsm) {
        vkDestroyShaderModule(sc.dev, sc.fsm, nullptr);
    }

    if (sc.vsm) {
        vkDestroyShaderModule(sc.dev, sc.vsm, nullptr);
    }

    destroy_swap_resources(sc);

    if (sc.dev) {
        vkDestroyDevice(sc.dev, nullptr);
    }

    if (sc.surf) {
        vkDestroySurfaceKHR(sc.inst, sc.surf, nullptr);
    }

    if (sc.inst) {
        vkDestroyInstance(sc.inst, nullptr);
    }
}

/* ================================================================ */
/*                       Main entry point                           */
/* ================================================================ */

void run_viewer(const ShowMesh& mesh, std::function<ShowTraj()> simulate,
                const ShowTraj* initial_traj = nullptr) {
    if (!glfwInit()) {
        fprintf(stderr, "[show] glfwInit failed\n");
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const int WIN_W = 1280, WIN_H = 720;
    const char* init_title = initial_traj ?
                             "rtmmc-vulkan | Trajectory viewer - rotate to inspect" :
                             "rtmmc-vulkan | SPACE = Start Simulation";
    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, init_title, nullptr, nullptr);

    if (!win) {
        fprintf(stderr, "[show] glfwCreateWindow failed\n");
        glfwTerminate();
        return;
    }

    InputState inp;
    inp.W = WIN_W;
    inp.H = WIN_H;
    glfwSetWindowUserPointer(win, &inp);
    glfwSetCursorPosCallback(win, cb_cursor);
    glfwSetMouseButtonCallback(win, cb_mouse_btn);
    glfwSetScrollCallback(win, cb_scroll);
    glfwSetKeyCallback(win, cb_key);
    glfwSetFramebufferSizeCallback(win, cb_resize);

    ShowCtx sc;
    init_viewer(sc, win, (uint32_t)WIN_W, (uint32_t)WIN_H);
    upload_mesh(sc, mesh);
    upload_src(sc, mesh.src_lines);
    upload_button(sc, 0);

    Camera cam;
    cam.cx = mesh.cx;
    cam.cy = mesh.cy;
    cam.cz = mesh.cz;
    cam.dist = mesh.radius * 2.5f;
    cam.W = WIN_W;
    cam.H = WIN_H;

    int state = initial_traj ? 2 : 0; /* 0=waiting, 1=running, 2=done */
    bool has_traj = (initial_traj != nullptr);
    ShowTraj traj;

    if (initial_traj) {
        traj = *initial_traj;
        upload_traj(sc, traj);
        upload_button(sc, 2);
    }

    g_frame_count = 0;

    auto clampf = [](float v, float lo, float hi) -> float {
        return v < lo ? lo : (v > hi ? hi : v);
    };

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        /* Handle window resize */
        if (inp.resized) {
            int fw = 0, fh = 0;
            glfwGetFramebufferSize(win, &fw, &fh);

            if (fw > 0 && fh > 0) {
                recreate_swap(sc, win);
                cam.W = fw;
                cam.H = fh;
            }

            inp.resized = false;
        }

        /* Apply camera orbit input */
        cam.yaw   += inp.dYaw;
        cam.pitch  = clampf(cam.pitch + inp.dPitch, -1.4f, 1.4f);
        cam.dist  *= (1.f - inp.dScroll * 0.1f);
        cam.dist   = clampf(cam.dist, mesh.radius * 0.1f, mesh.radius * 10.f);
        inp.dYaw = inp.dPitch = inp.dScroll = 0;

        /* Check if simulation should start */
        if ((inp.startSim || inp.clicked) && state == 0 && simulate) {
            state = 1;
            glfwSetWindowTitle(win, "rtmmc-vulkan | Running simulation...");
            upload_button(sc, 1);

            /* Show the "running" state before blocking */
            if (!draw_frame(sc, cam, state, has_traj)) {
                glfwSetWindowShouldClose(win, GLFW_TRUE);
                goto done;
            }

            g_frame_count++;
            vkQueueWaitIdle(sc.gq);

            /* Run the actual simulation (blocking) */
            traj = simulate();

            if (!traj.pts.empty()) {
                upload_traj(sc, traj);
                has_traj = true;
            }

            state = 2;
            upload_button(sc, 2);
            glfwSetWindowTitle(win, "rtmmc-vulkan | Simulation done - rotate to inspect");
        }

        inp.startSim = false;
        inp.clicked  = false;

        if (!draw_frame(sc, cam, state, has_traj)) {
            /* GPU device lost (watchdog). Close the window; trajectory was saved to disk. */
            glfwSetWindowShouldClose(win, GLFW_TRUE);
            continue;
        }

        g_frame_count++;
    }

done:
    vkDeviceWaitIdle(sc.dev);
    destroy_viewer(sc);
    glfwDestroyWindow(win);
    glfwTerminate();
}
