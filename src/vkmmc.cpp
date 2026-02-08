/*
 * vkmmc.cpp — Vulkan Ray Query MC photon transport (Single-AS)
 * Build:
 *   glslangValidator --target-env vulkan1.2 -e main -o vkmmc_core.spv vkmmc_core.comp
 *   g++ -std=c++11 -O2 -o vkmmc vkmmc.cpp miniz.c -lvulkan
 * Usage:
 *   ./vkmmc input.json
 *   ./vkmmc -f input.json -n 1e7 -B 2000000
 */
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include "vkmmc_io.h"
#include "vkmmc_shapes.h"
#include "vkmmc_curvature.h"

#ifndef VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    #define VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME "VK_EXT_shader_atomic_float"
#endif
#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT
    #define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT ((VkStructureType)1000260000)
#endif
#ifndef VK_EXT_shader_atomic_float
typedef struct VkPhysicalDeviceShaderAtomicFloatFeaturesEXT {
    VkStructureType sType;
    void* pNext;
    VkBool32 shaderBufferFloat32Atomics, shaderBufferFloat32AtomicAdd;
    VkBool32 shaderBufferFloat64Atomics, shaderBufferFloat64AtomicAdd;
    VkBool32 shaderSharedFloat32Atomics, shaderSharedFloat32AtomicAdd;
    VkBool32 shaderSharedFloat64Atomics, shaderSharedFloat64AtomicAdd;
    VkBool32 shaderImageFloat32Atomics, shaderImageFloat32AtomicAdd;
    VkBool32 sparseImageFloat32Atomics, sparseImageFloat32AtomicAdd;
} VkPhysicalDeviceShaderAtomicFloatFeaturesEXT;
#endif

/* ================================================================ */
/*                       MCParams uniform                           */
/* ================================================================ */
struct MCParams {
    /* vec4-aligned block (offset 0) */
    float    srcpos[4];                     //  0: vec4
    float    srcdir[4];                     // 16: vec4
    float    srcparam1[4];                  // 32: vec4
    float    srcparam2[4];                  // 48: vec4
    float    nmin[4];                       // 64: vec4  (grid_min)
    float    nmax[4];                       // 80: vec4  (grid_extent)
    uint32_t crop0[4];                      // 96: uvec4 (grid_stride)

    /* scalar block — packed 4 per row (offset 112) */
    float    dstep, tstart, tend, Rtstep;   // 112: 4 floats
    int      srctype, maxgate, outputtype;  // 128: 3 ints
    uint32_t isreflect;                     // 128+12: 1 uint  → 16 bytes
    uint32_t mediumid0, total_threads;      // 144: 2 uints
    uint32_t num_media, seed;               // 144+8: 2 uints → 16 bytes
    uint32_t do_csg, has_curvature;         // 160: 2 uints
    int      threadphoton, oddphoton;       // 160+8: 2 ints  → 16 bytes
};
// Total: 176 bytes

/* ================================================================ */
/*                       Vulkan helpers                             */
/* ================================================================ */
#define VK_CHECK(x) do{VkResult r=(x);if(r!=VK_SUCCESS){fprintf(stderr,"VK error %d at %s:%d\n",r,__FILE__,__LINE__);abort();}}while(0)

struct VulkanCtx {
    VkInstance instance;
    VkPhysicalDevice physDev;
    VkDevice device;
    uint32_t computeQF;
    VkQueue queue;
    VkCommandPool cmdPool;
    PFN_vkGetBufferDeviceAddressKHR pfnGetBufAddr;
    PFN_vkCreateAccelerationStructureKHR pfnCreateAS;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAS;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASBuildSizes;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASAddr;
};
struct Buffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    Buffer(): buf(VK_NULL_HANDLE), mem(VK_NULL_HANDLE), size(0) {}
};
struct AccelStruct {
    VkAccelerationStructureKHR handle;
    Buffer buffer;
    AccelStruct(): handle(VK_NULL_HANDLE) {}
};
struct ComputePipe {
    VkDescriptorSetLayout descLayout;
    VkPipelineLayout pipeLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
};

uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }

    throw std::runtime_error("no suitable memory type");
}

Buffer create_buffer(VulkanCtx& c, VkDeviceSize sz, VkBufferUsageFlags usage, VkMemoryPropertyFlags memP) {
    Buffer b;
    b.size = sz;
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = sz;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device, &ci, NULL, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c.device, b.buf, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_mem_type(c.physDev, req.memoryTypeBits, memP);
    VkMemoryAllocateFlagsInfo fl = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.pNext = NULL;

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        ai.pNext = &fl;
    }

    VK_CHECK(vkAllocateMemory(c.device, &ai, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(c.device, b.buf, b.mem, 0));
    return b;
}

void upload_host(VulkanCtx& c, Buffer& b, const void* data, VkDeviceSize sz) {
    void* p;
    VK_CHECK(vkMapMemory(c.device, b.mem, 0, sz, 0, &p));
    memcpy(p, data, sz);
    vkUnmapMemory(c.device, b.mem);
}

VkDeviceAddress get_addr(VulkanCtx& c, VkBuffer buf) {
    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buf;
    return c.pfnGetBufAddr(c.device, &info);
}

void destroy_buf(VulkanCtx& c, Buffer& b) {
    if (b.buf) {
        vkDestroyBuffer(c.device, b.buf, NULL);
    }

    if (b.mem) {
        vkFreeMemory(c.device, b.mem, NULL);
    }

    b = Buffer();
}

VkCommandBuffer begin_cmd(VulkanCtx& c) {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = c.cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &ai, &cmd));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void end_submit(VulkanCtx& c, VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    vkFreeCommandBuffers(c.device, c.cmdPool, 1, &cmd);
}

Buffer create_device_buffer(VulkanCtx& c, VkDeviceSize sz, VkBufferUsageFlags usage) {
    return create_buffer(c, sz, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void upload_to_device(VulkanCtx& c, Buffer& dst, const void* data, VkDeviceSize sz) {
    Buffer stg = create_buffer(c, sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload_host(c, stg, data, sz);
    VkCommandBuffer cmd = begin_cmd(c);
    VkBufferCopy r = {0, 0, sz};
    vkCmdCopyBuffer(cmd, stg.buf, dst.buf, 1, &r);
    end_submit(c, cmd);
    destroy_buf(c, stg);
}

void download_from_device(VulkanCtx& c, Buffer& src, void* data, VkDeviceSize sz) {
    Buffer stg = create_buffer(c, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer cmd = begin_cmd(c);
    VkBufferCopy r = {0, 0, sz};
    vkCmdCopyBuffer(cmd, src.buf, stg.buf, 1, &r);
    end_submit(c, cmd);
    void* p;
    VK_CHECK(vkMapMemory(c.device, stg.mem, 0, sz, 0, &p));
    memcpy(data, p, sz);
    vkUnmapMemory(c.device, stg.mem);
    destroy_buf(c, stg);
}

/* ================================================================ */
/*                       Vulkan init                                */
/* ================================================================ */
void list_vulkan_gpus() {
    VkInstance inst;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;

    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        printf("Failed to create Vulkan instance\n");
        return;
    }

    uint32_t cnt = 0;
    vkEnumeratePhysicalDevices(inst, &cnt, NULL);
    std::vector<VkPhysicalDevice> pds(cnt);
    vkEnumeratePhysicalDevices(inst, &cnt, pds.data());
    const char* dt[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    printf("========================== Vulkan GPU Devices ==========================\n");
    int did = 0;

    for (size_t j = 0; j < pds.size(); j++) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(pds[j], &pr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pds[j], &mp);
        uint32_t ec = 0;
        vkEnumerateDeviceExtensionProperties(pds[j], NULL, &ec, NULL);
        std::vector<VkExtensionProperties> exts(ec);
        vkEnumerateDeviceExtensionProperties(pds[j], NULL, &ec, exts.data());
        bool rq = false, as = false, af = false;

        for (size_t k = 0; k < exts.size(); k++) {
            if (!strcmp(exts[k].extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                rq = true;
            }

            if (!strcmp(exts[k].extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
                as = true;
            }

            if (!strcmp(exts[k].extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
                af = true;
            }
        }

        uint64_t tm = 0;

        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                tm += mp.memoryHeaps[h].size;
            }

        bool ok = rq && as;

        if (ok) {
            did++;
        }

        printf("============ GPU device ID %d [%zu of %u]: %s %s============\n", ok ? did : 0, j + 1, cnt, pr.deviceName, ok ? "" : "(not usable) ");
        printf(" Vendor ID      :\t0x%04x\n Device type    :\t%s\n Global memory  :\t%lu B\n", pr.vendorID, dt[pr.deviceType < 5 ? pr.deviceType : 0], (unsigned long)tm);
        printf(" Ray query      :\t%s\n Accel structure:\t%s\n Atomic float   :\t%s\n", rq ? "Yes" : "No", as ? "Yes" : "No", af ? "Yes" : "No");
    }

    vkDestroyInstance(inst, NULL);
}

VulkanCtx init_vulkan(int gpuid) {
    VulkanCtx c;
    memset(&c, 0, sizeof(c));
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &c.instance));
    uint32_t cnt = 0;
    vkEnumeratePhysicalDevices(c.instance, &cnt, NULL);
    std::vector<VkPhysicalDevice> pds(cnt);
    vkEnumeratePhysicalDevices(c.instance, &cnt, pds.data());
    std::vector<VkPhysicalDevice> usable;

    for (size_t j = 0; j < pds.size(); j++) {
        uint32_t ec = 0;
        vkEnumerateDeviceExtensionProperties(pds[j], NULL, &ec, NULL);
        std::vector<VkExtensionProperties> exts(ec);
        vkEnumerateDeviceExtensionProperties(pds[j], NULL, &ec, exts.data());
        bool ok[4] = {};

        for (size_t k = 0; k < exts.size(); k++) {
            if (!strcmp(exts[k].extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
                ok[0] = true;
            }

            if (!strcmp(exts[k].extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
                ok[1] = true;
            }

            if (!strcmp(exts[k].extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
                ok[2] = true;
            }

            if (!strcmp(exts[k].extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
                ok[3] = true;
            }
        }

        if (ok[0] && ok[1] && ok[2] && ok[3]) {
            usable.push_back(pds[j]);
        }
    }

    if (usable.empty()) {
        throw std::runtime_error("No GPU with ray query support");
    }

    if (gpuid >= 1 && gpuid <= (int)usable.size()) {
        c.physDev = usable[gpuid - 1];
    } else {
        for (size_t j = 0; j < usable.size(); j++) {
            VkPhysicalDeviceProperties p2;
            vkGetPhysicalDeviceProperties(usable[j], &p2);

            if (p2.vendorID == 0x10DE && p2.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                c.physDev = usable[j];
                break;
            }
        }

        if (!c.physDev) {
            c.physDev = usable[0];
        }
    }

    {
        VkPhysicalDeviceProperties p2;
        vkGetPhysicalDeviceProperties(c.physDev, &p2);
        printf("Selected GPU [%d/%d]: %s (vendor=0x%04x type=%d)\n", gpuid ? gpuid : 1, (int)usable.size(), p2.deviceName, p2.vendorID, p2.deviceType);
    }

    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.physDev, &qfc, NULL);
    std::vector<VkQueueFamilyProperties> qfs(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(c.physDev, &qfc, qfs.data());
    c.computeQF = UINT32_MAX;

    for (uint32_t i = 0; i < qfc; i++) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            c.computeQF = i;
            break;
        }

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.computeQF;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceProperties devPr;
    vkGetPhysicalDeviceProperties(c.physDev, &devPr);
    bool isNV = (devPr.vendorID == 0x10DE), hasAF = false;

    if (isNV) {
        uint32_t ec2 = 0;
        vkEnumerateDeviceExtensionProperties(c.physDev, NULL, &ec2, NULL);
        std::vector<VkExtensionProperties> ex2(ec2);
        vkEnumerateDeviceExtensionProperties(c.physDev, NULL, &ec2, ex2.data());

        for (size_t k = 0; k < ex2.size(); k++)
            if (!strcmp(ex2[k].extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
                hasAF = true;
            }
    }

    const char* devExts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                             VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
                            };
    int nExts = hasAF ? 5 : 4;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT saf;
    memset(&saf, 0, sizeof(saf));
    saf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    saf.shaderBufferFloat32AtomicAdd = VK_TRUE;
    VkPhysicalDeviceBufferDeviceAddressFeatures bda = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bda.bufferDeviceAddress = VK_TRUE;
    bda.pNext = hasAF ? (void*)&saf : NULL;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asf.accelerationStructure = VK_TRUE;
    asf.pNext = &bda;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqf.rayQuery = VK_TRUE;
    rqf.pNext = &asf;
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.bufferDeviceAddress = VK_TRUE;
    v12.pNext = &rqf;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nExts;
    dci.ppEnabledExtensionNames = devExts;
    dci.pNext = &v12;
    VK_CHECK(vkCreateDevice(c.physDev, &dci, NULL, &c.device));
    vkGetDeviceQueue(c.device, c.computeQF, 0, &c.queue);

    if (hasAF) {
        printf("GPU supports native atomicAdd(float)\n");
    }

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = c.computeQF;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(c.device, &cpi, NULL, &c.cmdPool));
    c.pfnGetBufAddr = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(c.device, "vkGetBufferDeviceAddressKHR");
    c.pfnCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(c.device, "vkCreateAccelerationStructureKHR");
    c.pfnDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(c.device, "vkDestroyAccelerationStructureKHR");
    c.pfnGetASBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(c.device, "vkGetAccelerationStructureBuildSizesKHR");
    c.pfnCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(c.device, "vkCmdBuildAccelerationStructuresKHR");
    c.pfnGetASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(c.device, "vkGetAccelerationStructureDeviceAddressKHR");
    return c;
}

/* ================================================================ */
/*                       Build TLAS                                 */
/* ================================================================ */
AccelStruct build_tlas(VulkanCtx& c, AccelStruct& blas) {
    VkAccelerationStructureDeviceAddressInfoKHR ai = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    ai.accelerationStructure = blas.handle;
    VkDeviceAddress ba = c.pfnGetASAddr(c.device, &ai);
    VkAccelerationStructureInstanceKHR inst;
    memset(&inst, 0, sizeof(inst));
    inst.transform.matrix[0][0] = 1.f;
    inst.transform.matrix[1][1] = 1.f;
    inst.transform.matrix[2][2] = 1.f;
    inst.mask = 0xFF;
    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst.accelerationStructureReference = ba;
    const VkBufferUsageFlags gu = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer ib = create_buffer(c, sizeof(inst), gu, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload_host(c, ib, &inst, sizeof(inst));
    VkAccelerationStructureGeometryInstancesDataKHR id = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    id.data.deviceAddress = get_addr(c, ib.buf);
    VkAccelerationStructureGeometryKHR geom = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = id;
    VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &geom;
    uint32_t ic = 1;
    VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    c.pfnGetASBuildSizes(c.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &ic, &sz);
    AccelStruct as;
    as.buffer = create_buffer(c, sz.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkAccelerationStructureCreateInfoKHR asci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asci.buffer = as.buffer.buf;
    asci.size = sz.accelerationStructureSize;
    asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    c.pfnCreateAS(c.device, &asci, NULL, &as.handle);
    Buffer scratch = create_buffer(c, sz.buildScratchSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bi.dstAccelerationStructure = as.handle;
    bi.scratchData.deviceAddress = get_addr(c, scratch.buf);
    VkAccelerationStructureBuildRangeInfoKHR range;
    memset(&range, 0, sizeof(range));
    range.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* pR = &range;
    VkCommandBuffer cmd = begin_cmd(c);
    c.pfnCmdBuildAS(cmd, 1, &bi, &pR);
    end_submit(c, cmd);
    destroy_buf(c, scratch);
    destroy_buf(c, ib);
    return as;
}

/* ================================================================ */
/*                       Compute pipeline                           */
/* ================================================================ */
std::vector<uint32_t> load_spirv(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);

    if (!f) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }

    size_t sz = (size_t)f.tellg();
    std::vector<uint32_t> code(sz / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), sz);
    return code;
}

ComputePipe create_pipeline(VulkanCtx& c, const char* spirv) {
    ComputePipe cp;
    memset(&cp, 0, sizeof(cp));
    VkDescriptorSetLayoutBinding bn[7];
    memset(bn, 0, sizeof(bn));
    bn[0].binding = 0;
    bn[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bn[0].descriptorCount = 1;
    bn[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    for (int i = 1; i <= 3; i++) {
        bn[i].binding = (uint32_t)i;
        bn[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bn[i].descriptorCount = 1;
        bn[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    bn[4].binding = 4;
    bn[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bn[4].descriptorCount = 1;
    bn[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bn[5].binding = 5;
    bn[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bn[5].descriptorCount = 1;
    bn[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bn[6].binding = 6;
    bn[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bn[6].descriptorCount = 1;
    bn[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 7;
    dl.pBindings = bn;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dl, NULL, &cp.descLayout));
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &cp.descLayout;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pl, NULL, &cp.pipeLayout));
    std::vector<uint32_t> code = load_spirv(spirv);
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = code.size() * 4;
    sm.pCode = code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(c.device, &sm, NULL, &mod));
    VkPipelineShaderStageCreateInfo st = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    st.module = mod;
    st.pName = "main";
    VkComputePipelineCreateInfo cp2 = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp2.stage = st;
    cp2.layout = cp.pipeLayout;
    VK_CHECK(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cp2, NULL, &cp.pipeline));
    vkDestroyShaderModule(c.device, mod, NULL);
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1;
    dp.poolSizeCount = 3;
    dp.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dp, NULL, &cp.descPool));
    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = cp.descPool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &cp.descLayout;
    VK_CHECK(vkAllocateDescriptorSets(c.device, &da, &cp.descSet));
    return cp;
}

void update_desc(VulkanCtx& c, ComputePipe& cp, VkAccelerationStructureKHR tlas,
                 Buffer& fb, Buffer& mb, Buffer& ob, Buffer& pb, Buffer& sb, Buffer& cb) {
    VkWriteDescriptorSetAccelerationStructureKHR aw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    aw.accelerationStructureCount = 1;
    aw.pAccelerationStructures = &tlas;
    VkDescriptorBufferInfo fi = {fb.buf, 0, VK_WHOLE_SIZE}, mi = {mb.buf, 0, VK_WHOLE_SIZE},
                           oi = {ob.buf, 0, VK_WHOLE_SIZE}, pi = {pb.buf, 0, VK_WHOLE_SIZE},
                           si = {sb.buf, 0, VK_WHOLE_SIZE}, ci = {cb.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[7];
    memset(w, 0, sizeof(w));
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].pNext = &aw;
    w[0].dstSet = cp.descSet;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    for (int i = 1; i <= 6; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = cp.descSet;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 4) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }

    w[1].pBufferInfo = &fi;
    w[2].pBufferInfo = &mi;
    w[3].pBufferInfo = &oi;
    w[4].pBufferInfo = &pi;
    w[5].pBufferInfo = &si;
    w[6].pBufferInfo = &ci;
    vkUpdateDescriptorSets(c.device, 7, w, 0, NULL);
}

/* ================================================================ */
/*                       CLI parsing                                */
/* ================================================================ */
struct CmdOverrides {
    uint64_t nphoton, batch_size;
    uint32_t rng_seed, totalthread;
    float unitinmm;
    int outputtype, isreflect, isnormalize, gpuid;
    bool listgpu, dumpjson, dumpmesh;
    int meshres;
    int docurv;   // -1 = auto (enable for CSG), 0 = off, 1 = on
    bool debugcurv;
    std::string session_id, json_str, inputfile;
    CmdOverrides(): nphoton(0), batch_size(UINT64_MAX), rng_seed(0), totalthread(0), unitinmm(0),
        outputtype(-1), isreflect(-1), isnormalize(-1), gpuid(0), listgpu(false), dumpjson(false), dumpmesh(false), meshres(0), docurv(-1), debugcurv(false) {}
};

void printhelp(const char* n) {
    printf("vkmmc - Vulkan ray-tracing accelerated mesh Monte Carlo\n"
           "Usage: %s input.json  OR  %s -f input.json [flags]\n\n"
           "Flags:\n -f/--input\tJSON file\n -n/--photon\tphoton number\n -s/--session\tsession name\n"
           " -u/--unitinmm\tvoxel size [1]\n -E/--seed\tRNG seed\n -O/--outputtype\tx:energy,f:flux,l:fluence\n"
           " -b/--reflect\tmismatch [1]\n -U/--normalize\t[1]\n -t/--thread\tGPU threads [65536]\n"
           " -B/--batch\tphotons/batch [500000, 0=no batch]\n -G/--gpuid\tdevice [1]\n"
           " -m/--meshres\tshape mesh resolution [24]\n -c/--curv\tcurvature [1=on,0=off,-1=auto]\n --debugcurv\texport curvature normal error map\n"
           " -L/--listgpu\tlist GPUs\n --dumpjson\tdump config\n --dumpmesh\tsave mesh JSON\n -h/--help\n", n, n);
    exit(0);
}

SimConfig parse_cmdline(int argc, char** argv, CmdOverrides& ovr) {
    std::string inputfile;

    if (argc < 2) {
        printhelp(argv[0]);
    }

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);

        if ((a == "-f" || a == "--input") && i + 1 < argc) {
            inputfile = argv[++i];
        } else if ((a == "-n" || a == "--photon") && i + 1 < argc) {
            ovr.nphoton = (uint64_t)atof(argv[++i]);
        } else if ((a == "-s" || a == "--session") && i + 1 < argc) {
            ovr.session_id = argv[++i];
        } else if ((a == "-u" || a == "--unitinmm") && i + 1 < argc) {
            ovr.unitinmm = (float)atof(argv[++i]);
        } else if ((a == "-E" || a == "--seed") && i + 1 < argc) {
            ovr.rng_seed = (uint32_t)atoi(argv[++i]);
        } else if ((a == "-O" || a == "--outputtype") && i + 1 < argc) {
            char c2 = argv[++i][0];
            ovr.outputtype = (c2 == 'f' ? 0 : c2 == 'l' ? 1 : 2);
        } else if ((a == "-b" || a == "--reflect") && i + 1 < argc) {
            ovr.isreflect = atoi(argv[++i]);
        } else if ((a == "-U" || a == "--normalize") && i + 1 < argc) {
            ovr.isnormalize = atoi(argv[++i]);
        } else if ((a == "-t" || a == "--thread") && i + 1 < argc) {
            ovr.totalthread = (uint32_t)atoi(argv[++i]);
        } else if ((a == "-B" || a == "--batch") && i + 1 < argc) {
            ovr.batch_size = (uint64_t)atof(argv[++i]);
        } else if ((a == "-G" || a == "--gpuid") && i + 1 < argc) {
            ovr.gpuid = atoi(argv[++i]);
        } else if (a == "-L" || a == "--listgpu") {
            ovr.listgpu = true;
        } else if ((a == "-j" || a == "--json") && i + 1 < argc) {
            ovr.json_str = argv[++i];
        } else if (a == "--dumpjson") {
            ovr.dumpjson = true;
        } else if (a == "--dumpmesh") {
            ovr.dumpmesh = true;
        } else if (a == "--debugcurv") {
            ovr.debugcurv = true;
        } else if ((a == "-m" || a == "--meshres") && i + 1 < argc) {
            ovr.meshres = atoi(argv[++i]);
        } else if ((a == "-c" || a == "--curv") && i + 1 < argc) {
            ovr.docurv = atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            printhelp(argv[0]);
        } else if (a[0] != '-' && inputfile.empty()) {
            inputfile = a;
        }
    }

    if (inputfile.empty()) {
        fprintf(stderr, "No input JSON\n");
        printhelp(argv[0]);
    }

    ovr.inputfile = inputfile;
    SimConfig cfg = load_json_input(inputfile.c_str());

    if (ovr.nphoton > 0) {
        cfg.nphoton = ovr.nphoton;
    }

    if (ovr.rng_seed > 0) {
        cfg.rng_seed = ovr.rng_seed;
    }

    if (!ovr.session_id.empty()) {
        cfg.session_id = ovr.session_id;
    }

    if (ovr.unitinmm > 0) {
        cfg.unitinmm = ovr.unitinmm;
    }

    if (ovr.outputtype >= 0) {
        cfg.output_type = ovr.outputtype;
    }

    if (ovr.isreflect >= 0) {
        cfg.do_mismatch = (ovr.isreflect != 0);
    }

    if (ovr.isnormalize >= 0) {
        cfg.do_normalize = (ovr.isnormalize != 0);
    }

    if (ovr.meshres > 0) {
        cfg.mesh_res = ovr.meshres;
    }

    return cfg;
}

/* ================================================================ */
/*                             Main                                 */
/* ================================================================ */
int main(int argc, char** argv) {
    const char* spirvFile = "vkmmc_core.spv";

    if (argc > 2) {
        std::string la(argv[argc - 1]);

        if (la.size() > 4 && la.substr(la.size() - 4) == ".spv") {
            spirvFile = argv[--argc];
        }
    }

    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);

        if (a == "-L" || a == "--listgpu") {
            list_vulkan_gpus();
            return 0;
        }
    }

    CmdOverrides ovr;
    SimConfig cfg = parse_cmdline(argc, argv, ovr);

    /* ---- Phase 1: Generate CSG mesh if needed ---- */
    std::vector<NodeCurvature> curvData;
    bool has_curvature = false;
    bool want_curvature = (ovr.docurv < 0) ? true : (ovr.docurv != 0);  // auto = on for CSG

    if (cfg.is_csg) {
        std::ifstream sf(ovr.inputfile.c_str());
        json jroot;
        sf >> jroot;

        if (jroot.contains("Shapes") && jroot["Shapes"].is_array()) {
            float ext[6] = {0, 60, 0, 60, 0, 60};
            ShapeMesh sm = parse_shapes(jroot["Shapes"], ext, cfg.mesh_res);
            cfg.nodes = sm.nodes;
            cfg.faces = sm.faces;
            cfg.facedata = sm.facedata;
            cfg.face_shape_id = sm.shape_id;
            update_bbox(cfg);

            if (want_curvature) {
                std::vector<ShapeOrigin> shape_origins = extract_shape_origins(jroot["Shapes"]);
                curvData = compute_curvature(sm, shape_origins);
                has_curvature = true;
                printf("Curvature: computed for %zu nodes, %zu shapes\n",
                       curvData.size(), shape_origins.size());
            } else {
                printf("Curvature: disabled\n");
            }
        }

        if (cfg.nodes.empty()) {
            fprintf(stderr, "CSG: no shapes\n");
            return 1;
        }
    }

    /* ---- Phase 2: Handle dump commands ---- */
    if (ovr.dumpjson) {
        printf("{\"Session\":{\"ID\":\"%s\",\"Photons\":%lu},\"Mesh\":{\"Nodes\":%zu,\"Faces\":%zu}}\n",
               cfg.session_id.c_str(), (unsigned long)cfg.nphoton, cfg.nodes.size(), cfg.faces.size());
        return 0;
    }

    if (ovr.dumpmesh) {
        if (cfg.nodes.empty() || cfg.faces.empty()) {
            fprintf(stderr, "No mesh to dump\n");
            return 1;
        }

        printf("Verifying face data (%zu faces):\n", cfg.faces.size());

        for (size_t i = 0; i < cfg.faces.size() && i < 5; i++) {
            uint32_t pk = 0;
            memcpy(&pk, &cfg.facedata[i].packed_media, 4);
            printf("  face[%zu]: v=(%u,%u,%u) n=(%.3f,%.3f,%.3f) front=%u back=%u\n",
                   i, cfg.faces[i][0], cfg.faces[i][1], cfg.faces[i][2],
                   cfg.facedata[i].nx, cfg.facedata[i].ny, cfg.facedata[i].nz, pk >> 16, pk & 0xFFFF);
        }

        json dump;
        dump["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}};
        dump["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}};
        dump["Domain"]["LengthUnit"] = cfg.unitinmm;
        dump["Domain"]["Media"] = json::array();

        for (size_t i = 0; i < cfg.media.size(); i++)
            dump["Domain"]["Media"].push_back({{"mua", cfg.media[i].mua / cfg.unitinmm}, {"mus", cfg.media[i].mus / cfg.unitinmm}, {"g", cfg.media[i].g}, {"n", cfg.media[i].n}});
        dump["Optode"]["Source"]["Type"] = (cfg.srctype == 0 ? "pencil" : (cfg.srctype == 4 ? "planar" : (cfg.srctype == 8 ? "disk" : "unknown")));
        dump["Optode"]["Source"]["Pos"] = {cfg.srcpos[0], cfg.srcpos[1], cfg.srcpos[2]};
        dump["Optode"]["Source"]["Dir"] = {cfg.srcdir[0], cfg.srcdir[1], cfg.srcdir[2], cfg.srcdir[3]};
        size_t nn = cfg.nodes.size(), nf = cfg.faces.size();
        std::vector<float> nd(nn * 3);

        for (size_t i = 0; i < nn; i++) {
            nd[i * 3] = cfg.nodes[i].x;
            nd[i * 3 + 1] = cfg.nodes[i].y;
            nd[i * 3 + 2] = cfg.nodes[i].z;
        }

        std::vector<size_t> ndim = {nn, 3};
        dump["Shapes"]["MeshNode"] = jdata_encode("single", ndim, nd.data(), nn * 3 * sizeof(float), false);
        std::vector<int32_t> sd(nf * 4);

        for (size_t i = 0; i < nf; i++) {
            sd[i * 4] = (int32_t)(cfg.faces[i][0] + 1);
            sd[i * 4 + 1] = (int32_t)(cfg.faces[i][1] + 1);
            sd[i * 4 + 2] = (int32_t)(cfg.faces[i][2] + 1);

            if (i < cfg.face_shape_id.size()) {
                sd[i * 4 + 3] = (int32_t)cfg.face_shape_id[i];
            } else {
                uint32_t pk = 0;
                memcpy(&pk, &cfg.facedata[i].packed_media, 4);
                sd[i * 4 + 3] = (int32_t)(pk & 0xFFFF);
            }
        }

        std::vector<size_t> sdim = {nf, 4};
        dump["Shapes"]["MeshSurf"] = jdata_encode("int32", sdim, sd.data(), nf * 4 * sizeof(int32_t), false);
        std::string mf = cfg.session_id + "_mesh.json";
        std::ofstream of(mf.c_str());
        of << dump.dump(2) << std::endl;
        printf("Mesh saved to %s (%zu nodes, %zu faces, %s mode)\n", mf.c_str(), nn, nf, cfg.is_csg ? "CSG" : "mesh");
        return 0;
    }

    if (ovr.debugcurv) {
        if (!has_curvature || curvData.empty()) {
            fprintf(stderr, "debugcurv requires curvature data (use -c 1)\n");
            return 1;
        }

        // Re-read JSON for shape origins
        std::ifstream dcf(ovr.inputfile.c_str());
        json dcroot;
        dcf >> dcroot;
        std::vector<ShapeOrigin> origins = extract_shape_origins(dcroot["Shapes"]);

        // ---- Coarse mesh: already built (cfg.nodes, cfg.faces, curvData) ----
        printf("debugcurv: coarse mesh = %zu nodes, %zu faces (mesh_res=%d)\n",
               cfg.nodes.size(), cfg.faces.size(), cfg.mesh_res);

        // ---- Fine mesh: rebuild at higher resolution ----
        int fine_res = cfg.mesh_res * 4;
        float ext_fine[6] = {0, 60, 0, 60, 0, 60};
        ShapeMesh fine_mesh = parse_shapes(dcroot["Shapes"], ext_fine, fine_res);
        printf("debugcurv: fine mesh = %zu nodes, %zu faces (mesh_res=%d)\n",
               fine_mesh.nodes.size(), fine_mesh.faces.size(), fine_res);

        // ---- For each fine mesh node, find enclosing coarse triangle ----
        // Build per-shape coarse face lists for faster lookup
        // shape_id is 1-based in cfg.face_shape_id

        size_t nn_fine = fine_mesh.nodes.size();
        size_t nf_fine = fine_mesh.faces.size();

        // Compute analytic normal at each fine node
        // Also compute curvature-predicted normal using coarse mesh
        // node_shape mapping for fine mesh
        std::vector<int> fine_node_shape(nn_fine, -1);

        for (size_t fi = 0; fi < fine_mesh.faces.size(); fi++) {
            int si = (fi < fine_mesh.shape_id.size()) ? (int)fine_mesh.shape_id[fi] - 1 : -1;

            for (int k = 0; k < 3; k++) {
                uint32_t ni = fine_mesh.faces[fi][k];

                if (ni < nn_fine && fine_node_shape[ni] < 0) {
                    fine_node_shape[ni] = si;
                }
            }
        }

        // Output: x, y, z, curv_error, flat_error per fine node
        std::vector<float> out_nodes(nn_fine * 5);
        double curv_err_sum = 0, curv_err_max = 0;
        double flat_err_sum = 0, flat_err_max = 0;
        int counted = 0;

        // Helper: compute analytic normal at a point for a given shape
        auto analytic_normal = [&](float px, float py, float pz, int si, Vec3 fallback) -> Vec3 {
            Vec3 N = fallback;

            if (si >= 0 && si < (int)origins.size()) {
                const ShapeOrigin& so = origins[si];

                if (so.type == 1) { // sphere
                    float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
                    float dl = std::sqrt(dx * dx + dy * dy + dz * dz);

                    if (dl > 1e-10f) {
                        N.x = dx / dl;
                        N.y = dy / dl;
                        N.z = dz / dl;
                    }
                } else if (so.type == 2) { // cylinder
                    float dx = px - so.cx, dy = py - so.cy, dz = pz - so.cz;
                    float proj = dx * so.ax + dy * so.ay + dz * so.az;
                    float rx = dx - proj * so.ax, ry = dy - proj * so.ay, rz = dz - proj * so.az;
                    float rl = std::sqrt(rx * rx + ry * ry + rz * rz);

                    if (rl > 1e-10f) {
                        N.x = rx / rl;
                        N.y = ry / rl;
                        N.z = rz / rl;
                    }
                }
            }

            return N;
        };

        // Helper: compute barycentric coords of point P in triangle (A, B, C)
        // Returns true if P is inside, sets u, v, w (barycentric weights)
        auto barycentric = [](Vec3 P, Vec3 A, Vec3 B, Vec3 C, float & u, float & v, float & w) -> bool {
            Vec3 v0 = {B.x - A.x, B.y - A.y, B.z - A.z};
            Vec3 v1 = {C.x - A.x, C.y - A.y, C.z - A.z};
            Vec3 v2 = {P.x - A.x, P.y - A.y, P.z - A.z};
            float d00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
            float d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
            float d11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
            float d20 = v2.x * v0.x + v2.y * v0.y + v2.z * v0.z;
            float d21 = v2.x * v1.x + v2.y * v1.y + v2.z * v1.z;
            float denom = d00 * d11 - d01 * d01;

            if (std::fabs(denom) < 1e-20f) {
                return false;
            }

            v = (d11 * d20 - d01 * d21) / denom;
            w = (d00 * d21 - d01 * d20) / denom;
            u = 1.0f - v - w;
            return (u >= -0.01f && v >= -0.01f && w >= -0.01f);
        };

        // Helper: curvature-predicted normal at point (sx,sy,sz) using coarse triangle fi
        auto curv_normal_at = [&](float sx, float sy, float sz, size_t fi) -> Vec3 {
            uint32_t cv0 = cfg.faces[fi][0], cv1 = cfg.faces[fi][1], cv2 = cfg.faces[fi][2];
            float NB[3] = {0, 0, 0};
            uint32_t vidx[3] = {cv0, cv1, cv2};

            for (int vi = 0; vi < 3; vi++) {
                uint32_t idx = vidx[vi];
                float Ni[3] = {curvData[idx].nx, curvData[idx].ny, curvData[idx].nz};
                float k1 = curvData[idx].k1, k2 = curvData[idx].k2;
                float ui[3] = {curvData[idx].px, curvData[idx].py, curvData[idx].pz};
                float Pi[3] = {cfg.nodes[idx].x, cfg.nodes[idx].y, cfg.nodes[idx].z};
                // v = cross(u, N) — same as shader
                float vdir[3] = {
                    ui[1]* Ni[2] - ui[2]* Ni[1], ui[2]* Ni[0] - ui[0]* Ni[2], ui[0]* Ni[1] - ui[1]* Ni[0]
                };
                float di[3] = {sx - Pi[0], sy - Pi[1], sz - Pi[2]};

                // Use full displacement as in the paper
                float du = di[0] * ui[0] + di[1] * ui[1] + di[2] * ui[2];
                float dv = di[0] * vdir[0] + di[1] * vdir[1] + di[2] * vdir[2];
                NB[0] += Ni[0] + k1 * du * ui[0] + k2 * dv * vdir[0];
                NB[1] += Ni[1] + k1 * du * ui[1] + k2 * dv * vdir[1];
                NB[2] += Ni[2] + k1 * du * ui[2] + k2 * dv * vdir[2];
            }

            float nbl = std::sqrt(NB[0] * NB[0] + NB[1] * NB[1] + NB[2] * NB[2]);

            if (nbl > 1e-10f) {
                NB[0] /= nbl;
                NB[1] /= nbl;
                NB[2] /= nbl;
            }

            // Ensure consistent with coarse face normal
            float fn[3] = {cfg.facedata[fi].nx, cfg.facedata[fi].ny, cfg.facedata[fi].nz};

            if (NB[0]*fn[0] + NB[1]*fn[1] + NB[2]*fn[2] < 0) {
                NB[0] = -NB[0];
                NB[1] = -NB[1];
                NB[2] = -NB[2];
            }

            return {NB[0], NB[1], NB[2]};
        };

        // For each fine node, find closest coarse triangle by projecting onto
        // the triangle plane and checking containment. Use the triangle whose
        // centroid is nearest as fallback.
        printf("debugcurv: computing normal errors...\n");

        for (size_t ni = 0; ni < nn_fine; ni++) {
            Vec3 P = fine_mesh.nodes[ni];
            int si = fine_node_shape[ni];
            out_nodes[ni * 5 + 0] = P.x;
            out_nodes[ni * 5 + 1] = P.y;
            out_nodes[ni * 5 + 2] = P.z;
            out_nodes[ni * 5 + 3] = 0;
            out_nodes[ni * 5 + 4] = 0;

            Vec3 fallback = {0, 0, 1};
            Vec3 N_analytic = analytic_normal(P.x, P.y, P.z, si, fallback);

            // Find nearest coarse triangle centroid of same shape
            int best_fi = -1;
            float best_cdist = 1e30f;

            for (size_t fi = 0; fi < cfg.faces.size(); fi++) {
                int csi = (fi < cfg.face_shape_id.size()) ? (int)cfg.face_shape_id[fi] - 1 : -1;

                if (csi != si) {
                    continue;
                }

                Vec3 A = cfg.nodes[cfg.faces[fi][0]];
                Vec3 B = cfg.nodes[cfg.faces[fi][1]];
                Vec3 C = cfg.nodes[cfg.faces[fi][2]];
                float cx2 = (A.x + B.x + C.x) / 3, cy2 = (A.y + B.y + C.y) / 3, cz2 = (A.z + B.z + C.z) / 3;
                float d2 = (P.x - cx2) * (P.x - cx2) + (P.y - cy2) * (P.y - cy2) + (P.z - cz2) * (P.z - cz2);

                if (d2 < best_cdist) {
                    best_cdist = d2;
                    best_fi = (int)fi;
                }
            }

            // Refine: among the closest few triangles, pick the one where
            // the point projects best (smallest barycentric clamp distance)
            if (best_fi >= 0) {
                // Also check the triangles sharing vertices with best_fi
                // For simplicity, check all same-shape triangles within 2x centroid distance
                float search_r2 = best_cdist * 4.0f + 1.0f;
                int refined_fi = best_fi;
                float best_bary_err = 1e30f;

                for (size_t fi = 0; fi < cfg.faces.size(); fi++) {
                    int csi = (fi < cfg.face_shape_id.size()) ? (int)cfg.face_shape_id[fi] - 1 : -1;

                    if (csi != si) {
                        continue;
                    }

                    Vec3 A = cfg.nodes[cfg.faces[fi][0]];
                    Vec3 B = cfg.nodes[cfg.faces[fi][1]];
                    Vec3 C = cfg.nodes[cfg.faces[fi][2]];
                    float cx2 = (A.x + B.x + C.x) / 3, cy2 = (A.y + B.y + C.y) / 3, cz2 = (A.z + B.z + C.z) / 3;
                    float d2 = (P.x - cx2) * (P.x - cx2) + (P.y - cy2) * (P.y - cy2) + (P.z - cz2) * (P.z - cz2);

                    if (d2 > search_r2) {
                        continue;
                    }

                    float u, v, w;

                    if (barycentric(P, A, B, C, u, v, w)) {
                        // Clamp distance from valid range
                        float berr = 0;

                        if (u < 0) {
                            berr += u * u;
                        }

                        if (v < 0) {
                            berr += v * v;
                        }

                        if (w < 0) {
                            berr += w * w;
                        }

                        if (berr < best_bary_err) {
                            best_bary_err = berr;
                            refined_fi = (int)fi;
                        }
                    }
                }

                best_fi = refined_fi;
            }

            if (best_fi >= 0) {
                // Curvature-predicted normal
                Vec3 N_curv = curv_normal_at(P.x, P.y, P.z, (size_t)best_fi);

                // Ensure analytic normal consistent direction
                float fn[3] = {cfg.facedata[best_fi].nx, cfg.facedata[best_fi].ny, cfg.facedata[best_fi].nz};

                if (N_analytic.x * fn[0] + N_analytic.y * fn[1] + N_analytic.z * fn[2] < 0) {
                    N_analytic.x = -N_analytic.x;
                    N_analytic.y = -N_analytic.y;
                    N_analytic.z = -N_analytic.z;
                }

                // Flat face normal of coarse triangle
                Vec3 N_flat = {cfg.facedata[best_fi].nx, cfg.facedata[best_fi].ny, cfg.facedata[best_fi].nz};

                float curv_dot = N_curv.x * N_analytic.x + N_curv.y * N_analytic.y + N_curv.z * N_analytic.z;
                float flat_dot = N_flat.x * N_analytic.x + N_flat.y * N_analytic.y + N_flat.z * N_analytic.z;
                float cerr = 1.0f - std::min(1.0f, std::max(-1.0f, curv_dot));
                float ferr = 1.0f - std::min(1.0f, std::max(-1.0f, flat_dot));

                // Debug: print details for high-error nodes
                static int dbg_printed = 0;

                if (cerr > 0.5f && dbg_printed < 5) {
                    dbg_printed++;
                    Vec3 A = cfg.nodes[cfg.faces[best_fi][0]];
                    Vec3 B = cfg.nodes[cfg.faces[best_fi][1]];
                    Vec3 C = cfg.nodes[cfg.faces[best_fi][2]];
                    int csi = (best_fi < (int)cfg.face_shape_id.size()) ? (int)cfg.face_shape_id[best_fi] - 1 : -1;
                    printf("  HIGH ERR node %zu: pos=(%.2f,%.2f,%.2f) shape=%d\n", ni, P.x, P.y, P.z, si);
                    printf("    matched coarse face %d: shape=%d\n", best_fi, csi);
                    printf("    coarse verts: (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f) (%.2f,%.2f,%.2f)\n",
                           A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z);
                    printf("    N_curv=(%.4f,%.4f,%.4f) N_anal=(%.4f,%.4f,%.4f) N_flat=(%.4f,%.4f,%.4f)\n",
                           N_curv.x, N_curv.y, N_curv.z, N_analytic.x, N_analytic.y, N_analytic.z,
                           N_flat.x, N_flat.y, N_flat.z);
                    printf("    curv_dot=%.6f flat_dot=%.6f cerr=%.6f ferr=%.6f\n", curv_dot, flat_dot, cerr, ferr);

                    // Print vertex curvature data
                    for (int vv = 0; vv < 3; vv++) {
                        uint32_t idx = cfg.faces[best_fi][vv];
                        printf("    vtx %u: N=(%.3f,%.3f,%.3f) k1=%.4f k2=%.4f u=(%.3f,%.3f,%.3f)\n",
                               idx, curvData[idx].nx, curvData[idx].ny, curvData[idx].nz,
                               curvData[idx].k1, curvData[idx].k2,
                               curvData[idx].px, curvData[idx].py, curvData[idx].pz);
                    }
                }

                out_nodes[ni * 5 + 3] = cerr;
                out_nodes[ni * 5 + 4] = ferr;
                curv_err_sum += cerr;

                if (cerr > curv_err_max) {
                    curv_err_max = cerr;
                }

                flat_err_sum += ferr;

                if (ferr > flat_err_max) {
                    flat_err_max = ferr;
                }

                counted++;
            }
        }

        printf("debugcurv: %d nodes evaluated\n", counted);
        printf("  curvature normal error — mean=%.6e  max=%.6e\n", curv_err_sum / counted, curv_err_max);
        printf("  flat normal error      — mean=%.6e  max=%.6e\n", flat_err_sum / counted, flat_err_max);
        printf("  improvement ratio      — mean=%.2fx  max=%.2fx\n",
               (flat_err_sum / counted) / (curv_err_sum / counted + 1e-30),
               flat_err_max / (curv_err_max + 1e-30));

        // Print worst-error nodes
        printf("  worst curvature-error nodes:\n");
        std::vector<std::pair<float, size_t>> errs;

        for (size_t i = 0; i < nn_fine; i++)
            errs.push_back({out_nodes[i * 5 + 3], i});
        std::sort(errs.begin(), errs.end(), [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
            return a.first > b.first;
        });

        for (int i = 0; i < std::min(10, (int)errs.size()); i++) {
            size_t ni = errs[i].second;
            printf("    node %zu: pos=(%.2f,%.2f,%.2f) curv_err=%.6f flat_err=%.6f\n",
                   ni, out_nodes[ni * 5], out_nodes[ni * 5 + 1], out_nodes[ni * 5 + 2],
                   out_nodes[ni * 5 + 3], out_nodes[ni * 5 + 4]);
        }

        // Export JData JSON
        json dump;
        dump["Session"] = {{"ID", cfg.session_id + "_curv"}};
        dump["CoarseMesh"] = {{"Nodes", (int)cfg.nodes.size()}, {"Faces", (int)cfg.faces.size()}, {"MeshRes", cfg.mesh_res}};
        dump["FineMesh"] = {{"Nodes", (int)nn_fine}, {"Faces", (int)nf_fine}, {"MeshRes", fine_res}};

        // MeshNode: Nx5 (x, y, z, curv_error, flat_error)
        std::vector<size_t> ndim = {nn_fine, 5};
        dump["Shapes"]["MeshNode"] = jdata_encode("single", ndim,
                                     out_nodes.data(), nn_fine * 5 * sizeof(float), false);

        // MeshSurf from fine mesh: Mx4 (v1, v2, v3, shape_id) — 1-based
        std::vector<int32_t> surf_data(nf_fine * 4);

        for (size_t fi = 0; fi < nf_fine; fi++) {
            surf_data[fi * 4 + 0] = (int32_t)(fine_mesh.faces[fi][0] + 1);
            surf_data[fi * 4 + 1] = (int32_t)(fine_mesh.faces[fi][1] + 1);
            surf_data[fi * 4 + 2] = (int32_t)(fine_mesh.faces[fi][2] + 1);
            surf_data[fi * 4 + 3] = (fi < fine_mesh.shape_id.size()) ? (int32_t)fine_mesh.shape_id[fi] : 1;
        }

        std::vector<size_t> sdim = {nf_fine, 4};
        dump["Shapes"]["MeshSurf"] = jdata_encode("int32", sdim,
                                     surf_data.data(), nf_fine * 4 * sizeof(int32_t), false);

        std::string outf = cfg.session_id + "_curv.json";
        std::ofstream ofs(outf.c_str());
        ofs << dump.dump(2) << std::endl;
        printf("debugcurv: saved to %s\n", outf.c_str());

        return 0;
    }

    /* ---- Phase 3: Vulkan simulation ---- */
    VulkanCtx ctx = init_vulkan(ovr.gpuid);

    /* Build BLAS */
    AccelStruct blas, tlas;
    {
        const VkBufferUsageFlags gu = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        Buffer vb = create_buffer(ctx, cfg.nodes.size() * sizeof(Vec3), gu, hv);
        upload_host(ctx, vb, cfg.nodes.data(), cfg.nodes.size()*sizeof(Vec3));
        std::vector<uint32_t> idx;
        idx.reserve(cfg.faces.size() * 3);

        for (size_t i = 0; i < cfg.faces.size(); i++) {
            idx.push_back(cfg.faces[i][0]);
            idx.push_back(cfg.faces[i][1]);
            idx.push_back(cfg.faces[i][2]);
        }

        Buffer ib = create_buffer(ctx, idx.size() * sizeof(uint32_t), gu, hv);
        upload_host(ctx, ib, idx.data(), idx.size()*sizeof(uint32_t));
        VkAccelerationStructureGeometryTrianglesDataKHR tr = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        tr.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tr.vertexData.deviceAddress = get_addr(ctx, vb.buf);
        tr.vertexStride = sizeof(Vec3);
        tr.maxVertex = (uint32_t)cfg.nodes.size() - 1;
        tr.indexType = VK_INDEX_TYPE_UINT32;
        tr.indexData.deviceAddress = get_addr(ctx, ib.buf);
        VkAccelerationStructureGeometryKHR geom = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tr;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geom;
        uint32_t pc = (uint32_t)cfg.faces.size();
        VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        ctx.pfnGetASBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &pc, &sz);
        blas.buffer = create_buffer(ctx, sz.accelerationStructureSize,
                                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR ac = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ac.buffer = blas.buffer.buf;
        ac.size = sz.accelerationStructureSize;
        ac.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        ctx.pfnCreateAS(ctx.device, &ac, NULL, &blas.handle);
        Buffer scratch = create_buffer(ctx, sz.buildScratchSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        bi.dstAccelerationStructure = blas.handle;
        bi.scratchData.deviceAddress = get_addr(ctx, scratch.buf);
        VkAccelerationStructureBuildRangeInfoKHR rng;
        memset(&rng, 0, sizeof(rng));
        rng.primitiveCount = pc;
        const VkAccelerationStructureBuildRangeInfoKHR* pR = &rng;
        VkCommandBuffer cmd = begin_cmd(ctx);
        ctx.pfnCmdBuildAS(cmd, 1, &bi, &pR);
        end_submit(ctx, cmd);
        destroy_buf(ctx, scratch);
        destroy_buf(ctx, vb);
        destroy_buf(ctx, ib);
        printf("BLAS: %u triangles, %lu KB\n", pc, (unsigned long)(sz.accelerationStructureSize / 1024));
        tlas = build_tlas(ctx, blas);
    }

    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    /* Face buffer: interleaved [normal+media, vertex_indices] per triangle */
    struct FaceDataGPU {
        float nx, ny, nz, packed_media;
        float v0f, v1f, v2f, pad;
    };
    std::vector<FaceDataGPU> faceGpu(cfg.faces.size());

    for (size_t i = 0; i < cfg.faces.size(); i++) {
        faceGpu[i].nx = cfg.facedata[i].nx;
        faceGpu[i].ny = cfg.facedata[i].ny;
        faceGpu[i].nz = cfg.facedata[i].nz;
        faceGpu[i].packed_media = cfg.facedata[i].packed_media;
        uint32_t v0 = cfg.faces[i][0], v1 = cfg.faces[i][1], v2 = cfg.faces[i][2];
        memcpy(&faceGpu[i].v0f, &v0, 4);
        memcpy(&faceGpu[i].v1f, &v1, 4);
        memcpy(&faceGpu[i].v2f, &v2, 4);
        faceGpu[i].pad = 0;
    }

    Buffer faceBuf = create_device_buffer(ctx, faceGpu.size() * sizeof(FaceDataGPU), ssbo);
    upload_to_device(ctx, faceBuf, faceGpu.data(), faceGpu.size()*sizeof(FaceDataGPU));

    /* Curvature buffer: 3 x vec4 per node */
    struct GpuNodeCurv {
        float nx, ny, nz, k1, px, py, pz, k2, posx, posy, posz, pad;
    };
    std::vector<GpuNodeCurv> gpuCurv;

    if (has_curvature && !curvData.empty()) {
        gpuCurv.resize(curvData.size());

        for (size_t i = 0; i < curvData.size(); i++) {
            gpuCurv[i] = {curvData[i].nx, curvData[i].ny, curvData[i].nz, curvData[i].k1,
                          curvData[i].px, curvData[i].py, curvData[i].pz, curvData[i].k2,
                          cfg.nodes[i].x, cfg.nodes[i].y, cfg.nodes[i].z, 0
                         };
        }
    } else {
        gpuCurv.resize(1);
        memset(&gpuCurv[0], 0, sizeof(GpuNodeCurv));
    }

    Buffer curvBuf = create_device_buffer(ctx, gpuCurv.size() * sizeof(GpuNodeCurv), ssbo);
    upload_to_device(ctx, curvBuf, gpuCurv.data(), gpuCurv.size()*sizeof(GpuNodeCurv));

    /* Media buffer */
    Buffer mediaBuf = create_device_buffer(ctx, cfg.media.size() * sizeof(Medium), ssbo);
    upload_to_device(ctx, mediaBuf, cfg.media.data(), cfg.media.size()*sizeof(Medium));

    /* Output grid */
    float vs = cfg.unitinmm;

    if (cfg.has_steps) {
        vs = cfg.steps[0];
    }

    float ge = vs * 0.5f;
    float gmin[3] = {cfg.nmin.x - ge, cfg.nmin.y - ge, cfg.nmin.z - ge};
    float gmax[3] = {cfg.nmax.x + ge, cfg.nmax.y + ge, cfg.nmax.z + ge};
    uint32_t nx = (uint32_t)ceil((gmax[0] - gmin[0]) / vs);
    uint32_t ny = (uint32_t)ceil((gmax[1] - gmin[1]) / vs);
    uint32_t nz = (uint32_t)ceil((gmax[2] - gmin[2]) / vs);
    uint32_t crop0w = nx * ny * nz * cfg.maxgate, outSz = crop0w * 2;

    Buffer outBuf = create_device_buffer(ctx, outSz * sizeof(float), ssbo);
    {
        VkCommandBuffer cmd = begin_cmd(ctx);
        vkCmdFillBuffer(cmd, outBuf.buf, 0, outSz * sizeof(float), 0);
        end_submit(ctx, cmd);
    }
    printf("Grid: %ux%ux%u x %d gates, voxel=%.3fmm, origin=[%.2f,%.2f,%.2f]\n",
           nx, ny, nz, cfg.maxgate, vs, gmin[0], gmin[1], gmin[2]);

    /* Threads */
    uint32_t tt = (ovr.totalthread > 0) ? ovr.totalthread : 65536;

    if (cfg.nphoton < tt) {
        tt = ((uint32_t)cfg.nphoton + 63) / 64 * 64;

        if (!tt) {
            tt = 64;
        }
    }

    /* MCParams */
    MCParams params;
    memset(&params, 0, sizeof(params));

    for (int i = 0; i < 3; i++) {
        params.srcpos[i] = cfg.srcpos[i];
        params.srcdir[i] = cfg.srcdir[i];
    }

    for (int i = 0; i < 4; i++) {
        params.srcparam1[i] = cfg.srcparam1[i];
        params.srcparam2[i] = cfg.srcparam2[i];
    }

    params.nmin[0] = gmin[0];
    params.nmin[1] = gmin[1];
    params.nmin[2] = gmin[2];
    params.nmax[0] = gmax[0] - gmin[0];
    params.nmax[1] = gmax[1] - gmin[1];
    params.nmax[2] = gmax[2] - gmin[2];
    params.crop0[0] = nx;
    params.crop0[1] = nx * ny;
    params.crop0[2] = nx * ny * nz;
    params.crop0[3] = crop0w;

    params.dstep = 1.0f / vs;
    params.tstart = cfg.t0;
    params.tend = cfg.t1;
    params.Rtstep = 1.0f / cfg.dt;

    params.srctype = cfg.srctype;
    params.maxgate = cfg.maxgate;
    params.outputtype = cfg.output_type;
    params.isreflect = cfg.do_mismatch ? 1u : 0u;

    params.mediumid0 = cfg.mediumid0;
    params.total_threads = tt;
    params.num_media = (uint32_t)cfg.media.size();
    params.seed = cfg.rng_seed;

    params.do_csg = cfg.is_csg ? 1u : 0u;
    params.has_curvature = has_curvature ? 1u : 0u;
    params.do_csg = cfg.is_csg ? 1u : 0u;
    params.has_curvature = has_curvature ? 1u : 0u;

    Buffer paramBuf = create_device_buffer(ctx, sizeof(MCParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    /* RNG seeds */
    srand(cfg.rng_seed > 0 ? cfg.rng_seed : (uint32_t)time(0));
    struct uint4_t {
        uint32_t x, y, z, w;
    };
    std::vector<uint4_t> seeds(tt);

    for (uint32_t i = 0; i < tt; i++) seeds[i] = {(uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand()};

    Buffer seedBuf = create_device_buffer(ctx, tt * sizeof(uint4_t), ssbo);

    upload_to_device(ctx, seedBuf, seeds.data(), tt * sizeof(uint4_t));

    if (has_curvature && !curvData.empty()) {
        gpuCurv.resize(curvData.size());

        for (size_t i = 0; i < curvData.size(); i++) {
            gpuCurv[i] = { curvData[i].nx, curvData[i].ny, curvData[i].nz, curvData[i].k1,
                           curvData[i].px, curvData[i].py, curvData[i].pz, curvData[i].k2
                         };
        }
    } else {
        gpuCurv.resize(1); // dummy
        memset(&gpuCurv[0], 0, sizeof(GpuNodeCurv));
    }

    ComputePipe cp = create_pipeline(ctx, spirvFile);

    update_desc(ctx, cp, tlas.handle, faceBuf, mediaBuf, outBuf, paramBuf, seedBuf, curvBuf);

    // Batched dispatch
    uint64_t batchsz;

    if (ovr.batch_size == 0) {
        batchsz = cfg.nphoton;
    } else if (ovr.batch_size != UINT64_MAX) {
        batchsz = ovr.batch_size;
    } else {
        batchsz = 500000;
    }

    uint64_t pdone = 0;
    int batch = 0;
    uint32_t wg = tt / 64;
    printf("Threads: %u (%u x 64), batch: %lu photons\n", tt, wg, (unsigned long)batchsz);

    typedef std::chrono::high_resolution_clock Clk;
    double kms = 0;
    Clk::time_point t0 = Clk::now();

    while (pdone < cfg.nphoton) {
        uint64_t rem = cfg.nphoton - pdone, bp = (rem < batchsz) ? rem : batchsz;
        params.threadphoton = (int)(bp / tt);
        params.oddphoton = (int)(bp - (uint64_t)params.threadphoton * tt);
        upload_to_device(ctx, paramBuf, &params, sizeof(params));
        VkCommandBuffer cmd = begin_cmd(ctx);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeLayout, 0, 1, &cp.descSet, 0, NULL);
        vkCmdDispatch(cmd, wg, 1, 1);
        Clk::time_point ks = Clk::now();
        end_submit(ctx, cmd);
        kms += std::chrono::duration<double, std::milli>(Clk::now() - ks).count();
        pdone += bp;
        batch++;
        printf("  batch %d: %lu photons (%lu/%lu)\n", batch, (unsigned long)bp, (unsigned long)pdone, (unsigned long)cfg.nphoton);
    }

    double tms = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
    printf("Done (%d batches), kernel: %.3f ms, total: %.3f ms\n", batch, kms, tms);
    printf("Speed: %.2f photon/ms (kernel), %.2f photon/ms (total)\n", (double)cfg.nphoton / kms, (double)cfg.nphoton / tms);

    // Readback
    std::vector<float> raw(outSz);
    download_from_device(ctx, outBuf, raw.data(), outSz * sizeof(float));

    std::vector<float> fluence(crop0w);
    double absorbed = 0;
    int nans = 0, infs = 0;

    for (uint32_t i = 0; i < crop0w; i++) {
        fluence[i] = raw[i] + raw[i + crop0w];

        if (fluence[i] != fluence[i]) {
            nans++;
            fluence[i] = 0;
        } else if (fluence[i] > 1e30f || fluence[i] < -1e30f) {
            infs++;
            fluence[i] = 0;
        } else {
            absorbed += fluence[i];
        }
    }

    if (nans || infs) {
        printf("WARNING: %d NaN, %d Inf voxels zeroed\n", nans, infs);
    }

    printf("simulated %lu photons, absorbed: %.5f%%\n", (unsigned long)cfg.nphoton, absorbed / (double)cfg.nphoton * 100.0);

    if (cfg.do_normalize) {
        float vv = vs * vs * vs;

        for (uint32_t i = 0; i < crop0w; i++) {
            fluence[i] /= (float)cfg.nphoton * vv;
        }
    }

    // Save JData
    {
        std::vector<size_t> dims;
        dims.push_back(nx);
        dims.push_back(ny);
        dims.push_back(nz);

        if (cfg.maxgate > 1) {
            dims.push_back((size_t)cfg.maxgate);
        }

        json root;
        root["Session"] = {{"ID", cfg.session_id}, {"Photons", cfg.nphoton}};
        root["Forward"] = {{"T0", cfg.t0}, {"T1", cfg.t1}, {"Dt", cfg.dt}};
        root["Domain"] = {{"LengthUnit", cfg.unitinmm}, {"VoxelSize", vs}, {"Dim", {nx, ny, nz}}, {"Origin", {gmin[0], gmin[1], gmin[2]}}};
        root["Fluence"] = jdata_encode("single", dims, fluence.data(), crop0w * sizeof(float));
        std::string outname = cfg.session_id + ".jdat";
        std::ofstream of(outname.c_str());
        of << root.dump(2) << std::endl;
        printf("Output: %s (%ux%ux%u", outname.c_str(), nx, ny, nz);

        if (cfg.maxgate > 1) {
            printf("x%d", cfg.maxgate);
        }

        printf(")\n");
    }

    // Cleanup
    ctx.pfnDestroyAS(ctx.device, tlas.handle, NULL);
    ctx.pfnDestroyAS(ctx.device, blas.handle, NULL);
    destroy_buf(ctx, tlas.buffer);
    destroy_buf(ctx, blas.buffer);
    destroy_buf(ctx, faceBuf);
    destroy_buf(ctx, mediaBuf);
    destroy_buf(ctx, outBuf);
    destroy_buf(ctx, paramBuf);
    destroy_buf(ctx, seedBuf);
    destroy_buf(ctx, curvBuf);
    vkDestroyPipeline(ctx.device, cp.pipeline, NULL);
    vkDestroyPipelineLayout(ctx.device, cp.pipeLayout, NULL);
    vkDestroyDescriptorPool(ctx.device, cp.descPool, NULL);
    vkDestroyDescriptorSetLayout(ctx.device, cp.descLayout, NULL);
    vkDestroyCommandPool(ctx.device, ctx.cmdPool, NULL);
    vkDestroyDevice(ctx.device, NULL);
    vkDestroyInstance(ctx.instance, NULL);
    return 0;
}