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

// ---- MCParams uniform (must match std140 in vkmmc_core.comp) ----
struct MCParams {
    int      srctype;           // 0
    int      _p0, _p1, _p2;    // 4,8,12
    float    srcpos[4];         // 16
    float    srcdir[4];         // 32
    float    srcparam1[4];      // 48
    float    srcparam2[4];      // 64
    float    nmin[4];           // 80
    float    nmax[4];           // 96
    uint32_t crop0[4];          // 112
    float    dstep;             // 128
    float    tstart, tend, Rtstep; // 132,136,140
    int      maxgate;           // 144
    uint32_t mediumid0;         // 148
    uint32_t isreflect;         // 152
    int      outputtype;        // 156
    int      threadphoton;      // 160
    int      oddphoton;         // 164
    uint32_t total_threads;     // 168
    uint32_t num_media;         // 172
    uint32_t seed;              // 176
    uint32_t _pad[3];           // 180 — pad to 192
};

// ---- Vulkan helpers ----
#define VK_CHECK(x) do{VkResult r=(x);if(r!=VK_SUCCESS){fprintf(stderr,"Vulkan error %d at %s:%d\n",r,__FILE__,__LINE__);abort();}}while(0)

struct VulkanCtx {
    VkInstance instance; VkPhysicalDevice physDev; VkDevice device;
    uint32_t computeQF; VkQueue queue; VkCommandPool cmdPool;
    PFN_vkGetBufferDeviceAddressKHR pfnGetBufAddr;
    PFN_vkCreateAccelerationStructureKHR pfnCreateAS;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAS;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASBuildSizes;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASAddr;
};
struct Buffer { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; Buffer():buf(VK_NULL_HANDLE),mem(VK_NULL_HANDLE),size(0){} };
struct AccelStruct { VkAccelerationStructureKHR handle; Buffer buffer; AccelStruct():handle(VK_NULL_HANDLE){} };
struct ComputePipe { VkDescriptorSetLayout descLayout; VkPipelineLayout pipeLayout; VkPipeline pipeline; VkDescriptorPool descPool; VkDescriptorSet descSet; };

uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((filter&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&props)==props) return i;
    throw std::runtime_error("no suitable memory type");
}

Buffer create_buffer(VulkanCtx& c, VkDeviceSize sz, VkBufferUsageFlags usage, VkMemoryPropertyFlags memP) {
    Buffer b; b.size=sz;
    VkBufferCreateInfo ci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device,&ci,NULL,&b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(c.device,b.buf,&req);
    VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=find_mem_type(c.physDev,req.memoryTypeBits,memP);
    VkMemoryAllocateFlagsInfo fl={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fl.pNext=NULL;
    if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT){fl.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;ai.pNext=&fl;}
    VkResult mr=vkAllocateMemory(c.device,&ai,NULL,&b.mem);
    if(mr!=VK_SUCCESS){fprintf(stderr,"vkAllocateMemory failed(%d) for %lu bytes\n",mr,(unsigned long)ai.allocationSize);abort();}
    VK_CHECK(vkBindBufferMemory(c.device,b.buf,b.mem,0));
    return b;
}

void upload_host(VulkanCtx& c, Buffer& b, const void* data, VkDeviceSize sz) {
    void* p; VK_CHECK(vkMapMemory(c.device,b.mem,0,sz,0,&p)); memcpy(p,data,sz); vkUnmapMemory(c.device,b.mem);
}

VkDeviceAddress get_addr(VulkanCtx& c, VkBuffer buf) {
    VkBufferDeviceAddressInfo info={VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer=buf;
    return c.pfnGetBufAddr(c.device,&info);
}

void destroy_buf(VulkanCtx& c, Buffer& b) {
    if(b.buf)vkDestroyBuffer(c.device,b.buf,NULL); if(b.mem)vkFreeMemory(c.device,b.mem,NULL); b=Buffer();
}

VkCommandBuffer begin_cmd(VulkanCtx& c) {
    VkCommandBufferAllocateInfo ai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool=c.cmdPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(c.device,&ai,&cmd));
    VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd,&bi)); return cmd;
}

void end_submit(VulkanCtx& c, VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    VK_CHECK(vkQueueSubmit(c.queue,1,&si,VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(c.queue));
    vkFreeCommandBuffers(c.device,c.cmdPool,1,&cmd);
}

// ---- Device-local buffer helpers (staging upload/download) ----
Buffer create_device_buffer(VulkanCtx& c, VkDeviceSize sz, VkBufferUsageFlags usage) {
    return create_buffer(c, sz, usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void upload_to_device(VulkanCtx& c, Buffer& dst, const void* data, VkDeviceSize sz) {
    Buffer stg=create_buffer(c,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    upload_host(c,stg,data,sz);
    VkCommandBuffer cmd=begin_cmd(c); VkBufferCopy r={0,0,sz}; vkCmdCopyBuffer(cmd,stg.buf,dst.buf,1,&r); end_submit(c,cmd);
    destroy_buf(c,stg);
}

void download_from_device(VulkanCtx& c, Buffer& src, void* data, VkDeviceSize sz) {
    Buffer stg=create_buffer(c,sz,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer cmd=begin_cmd(c); VkBufferCopy r={0,0,sz}; vkCmdCopyBuffer(cmd,src.buf,stg.buf,1,&r); end_submit(c,cmd);
    void* p; VK_CHECK(vkMapMemory(c.device,stg.mem,0,sz,0,&p)); memcpy(data,p,sz); vkUnmapMemory(c.device,stg.mem);
    destroy_buf(c,stg);
}

void zero_device_buffer(VulkanCtx& c, Buffer& dst, VkDeviceSize sz) {
    VkCommandBuffer cmd=begin_cmd(c); vkCmdFillBuffer(cmd,dst.buf,0,sz,0); end_submit(c,cmd);
}

// ---- Vulkan init ----
void list_vulkan_gpus() {
    VkInstance inst;
    VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    if(vkCreateInstance(&ici,NULL,&inst)!=VK_SUCCESS){printf("Failed to create Vulkan instance\n");return;}
    uint32_t cnt=0; vkEnumeratePhysicalDevices(inst,&cnt,NULL);
    std::vector<VkPhysicalDevice> pds(cnt); vkEnumeratePhysicalDevices(inst,&cnt,pds.data());
    const char* dtypes[]={"other","integrated","discrete","virtual","cpu"};
    printf("========================== Vulkan GPU Devices ==========================\n");
    int devid=0;
    for(size_t j=0;j<pds.size();j++){
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pds[j],&props);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pds[j],&mp);
        // Check ray query support
        uint32_t ec=0; vkEnumerateDeviceExtensionProperties(pds[j],NULL,&ec,NULL);
        std::vector<VkExtensionProperties> exts(ec); vkEnumerateDeviceExtensionProperties(pds[j],NULL,&ec,exts.data());
        bool hasRQ=false, hasAS=false, hasAF=false;
        for(size_t k=0;k<exts.size();k++){
            if(!strcmp(exts[k].extensionName,VK_KHR_RAY_QUERY_EXTENSION_NAME)) hasRQ=true;
            if(!strcmp(exts[k].extensionName,VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) hasAS=true;
            if(!strcmp(exts[k].extensionName,VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) hasAF=true;
        }
        uint64_t totalmem=0;
        for(uint32_t h=0;h<mp.memoryHeapCount;h++) if(mp.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) totalmem+=mp.memoryHeaps[h].size;
        bool usable=hasRQ&&hasAS;
        if(usable) devid++;
        printf("============ GPU device ID %d [%zu of %u]: %s %s============\n",usable?devid:0,j+1,cnt,props.deviceName,usable?"":"(not usable) ");
        printf(" Device %zu of %u:\t\t%s\n",j+1,cnt,props.deviceName);
        printf(" Vendor ID      :\t0x%04x\n",props.vendorID);
        printf(" Device type    :\t%s\n",dtypes[props.deviceType<5?props.deviceType:0]);
        printf(" API version    :\t%u.%u.%u\n",VK_API_VERSION_MAJOR(props.apiVersion),VK_API_VERSION_MINOR(props.apiVersion),VK_API_VERSION_PATCH(props.apiVersion));
        printf(" Global memory  :\t%lu B\n",(unsigned long)totalmem);
        printf(" Ray query      :\t%s\n",hasRQ?"Yes":"No");
        printf(" Accel structure:\t%s\n",hasAS?"Yes":"No");
        printf(" Atomic float   :\t%s\n",hasAF?"Yes":"No");
    }
    vkDestroyInstance(inst,NULL);
}

VulkanCtx init_vulkan(int gpuid) {
    VulkanCtx c; memset(&c,0,sizeof(c));
    VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VK_CHECK(vkCreateInstance(&ici,NULL,&c.instance));
    uint32_t cnt=0; vkEnumeratePhysicalDevices(c.instance,&cnt,NULL);
    std::vector<VkPhysicalDevice> pds(cnt); vkEnumeratePhysicalDevices(c.instance,&cnt,pds.data());
    for(size_t j=0;j<pds.size();j++){
        uint32_t ec=0; vkEnumerateDeviceExtensionProperties(pds[j],NULL,&ec,NULL);
        std::vector<VkExtensionProperties> exts(ec); vkEnumerateDeviceExtensionProperties(pds[j],NULL,&ec,exts.data());
        bool ok[4]={false,false,false,false};
        for(size_t k=0;k<exts.size();k++){
            if(!strcmp(exts[k].extensionName,VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))ok[0]=true;
            if(!strcmp(exts[k].extensionName,VK_KHR_RAY_QUERY_EXTENSION_NAME))ok[1]=true;
            if(!strcmp(exts[k].extensionName,VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))ok[2]=true;
            if(!strcmp(exts[k].extensionName,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME))ok[3]=true;
        }
        if(ok[0]&&ok[1]&&ok[2]&&ok[3]){c.physDev=pds[j];break;}
    }
    if(!c.physDev) throw std::runtime_error("No GPU with ray query support");
    uint32_t qfc=0; vkGetPhysicalDeviceQueueFamilyProperties(c.physDev,&qfc,NULL);
    std::vector<VkQueueFamilyProperties> qfs(qfc); vkGetPhysicalDeviceQueueFamilyProperties(c.physDev,&qfc,qfs.data());
    c.computeQF=UINT32_MAX; for(uint32_t i=0;i<qfc;i++) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){c.computeQF=i;break;}
    float prio=1.f; VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=c.computeQF; qci.queueCount=1; qci.pQueuePriorities=&prio;
    const char* devExts[]={VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,VK_KHR_RAY_QUERY_EXTENSION_NAME,VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES}; bda.bufferDeviceAddress=VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR}; asf.accelerationStructure=VK_TRUE; asf.pNext=&bda;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR}; rqf.rayQuery=VK_TRUE; rqf.pNext=&asf;
    VkPhysicalDeviceVulkan12Features v12={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; v12.bufferDeviceAddress=VK_TRUE; v12.pNext=&rqf;
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=4; dci.ppEnabledExtensionNames=devExts; dci.pNext=&v12;
    VK_CHECK(vkCreateDevice(c.physDev,&dci,NULL,&c.device)); vkGetDeviceQueue(c.device,c.computeQF,0,&c.queue);
    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=c.computeQF; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(c.device,&cpi,NULL,&c.cmdPool));
    c.pfnGetBufAddr=(PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(c.device,"vkGetBufferDeviceAddressKHR");
    c.pfnCreateAS=(PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(c.device,"vkCreateAccelerationStructureKHR");
    c.pfnDestroyAS=(PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(c.device,"vkDestroyAccelerationStructureKHR");
    c.pfnGetASBuildSizes=(PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(c.device,"vkGetAccelerationStructureBuildSizesKHR");
    c.pfnCmdBuildAS=(PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(c.device,"vkCmdBuildAccelerationStructuresKHR");
    c.pfnGetASAddr=(PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(c.device,"vkGetAccelerationStructureDeviceAddressKHR");
    return c;
}

// ---- Build TLAS ----
AccelStruct build_tlas(VulkanCtx& c, AccelStruct& blas) {
    VkAccelerationStructureDeviceAddressInfoKHR ai={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; ai.accelerationStructure=blas.handle;
    VkDeviceAddress blasAddr=c.pfnGetASAddr(c.device,&ai);
    VkAccelerationStructureInstanceKHR inst; memset(&inst,0,sizeof(inst));
    inst.transform.matrix[0][0]=1.f; inst.transform.matrix[1][1]=1.f; inst.transform.matrix[2][2]=1.f;
    inst.mask=0xFF; inst.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; inst.accelerationStructureReference=blasAddr;
    const VkBufferUsageFlags gu=VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const VkMemoryPropertyFlags hv=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer ib=create_buffer(c,sizeof(inst),gu,hv); upload_host(c,ib,&inst,sizeof(inst));
    VkAccelerationStructureGeometryInstancesDataKHR id={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR}; id.data.deviceAddress=get_addr(c,ib.buf);
    VkAccelerationStructureGeometryKHR geom={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geom.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR; geom.geometry.instances=id;
    VkAccelerationStructureBuildGeometryInfoKHR bi={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR; bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; bi.geometryCount=1; bi.pGeometries=&geom;
    uint32_t ic=1; VkAccelerationStructureBuildSizesInfoKHR sz={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    c.pfnGetASBuildSizes(c.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&bi,&ic,&sz);
    AccelStruct as; as.buffer=create_buffer(c,sz.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkAccelerationStructureCreateInfoKHR asci={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; asci.buffer=as.buffer.buf; asci.size=sz.accelerationStructureSize; asci.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    c.pfnCreateAS(c.device,&asci,NULL,&as.handle);
    Buffer scratch=create_buffer(c,sz.buildScratchSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bi.dstAccelerationStructure=as.handle; bi.scratchData.deviceAddress=get_addr(c,scratch.buf);
    VkAccelerationStructureBuildRangeInfoKHR range; memset(&range,0,sizeof(range)); range.primitiveCount=1;
    const VkAccelerationStructureBuildRangeInfoKHR* pR=&range;
    VkCommandBuffer cmd=begin_cmd(c); c.pfnCmdBuildAS(cmd,1,&bi,&pR); end_submit(c,cmd);
    destroy_buf(c,scratch); destroy_buf(c,ib); return as;
}

// ---- Compute pipeline ----
std::vector<uint32_t> load_spirv(const char* path) {
    std::ifstream f(path,std::ios::ate|std::ios::binary); if(!f) throw std::runtime_error(std::string("cannot open ")+path);
    size_t sz=(size_t)f.tellg(); std::vector<uint32_t> code(sz/4); f.seekg(0); f.read(reinterpret_cast<char*>(code.data()),sz); return code;
}

ComputePipe create_pipeline(VulkanCtx& c, const char* spirv) {
    ComputePipe cp; memset(&cp,0,sizeof(cp));
    VkDescriptorSetLayoutBinding bindings[6]; memset(bindings,0,sizeof(bindings));
    bindings[0].binding=0; bindings[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; bindings[0].descriptorCount=1; bindings[0].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    for(int i=1;i<=3;i++){bindings[i].binding=(uint32_t)i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    bindings[4].binding=4;bindings[4].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;bindings[4].descriptorCount=1;bindings[4].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[5].binding=5;bindings[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[5].descriptorCount=1;bindings[5].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dslci.bindingCount=6; dslci.pBindings=bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device,&dslci,NULL,&cp.descLayout));
    VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&cp.descLayout;
    VK_CHECK(vkCreatePipelineLayout(c.device,&plci,NULL,&cp.pipeLayout));
    std::vector<uint32_t> code=load_spirv(spirv);
    VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size()*4; smci.pCode=code.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(c.device,&smci,NULL,&sm));
    VkPipelineShaderStageCreateInfo stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.module=sm; stage.pName="main";
    VkComputePipelineCreateInfo cpci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=stage; cpci.layout=cp.pipeLayout;
    VK_CHECK(vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&cpci,NULL,&cp.pipeline));
    vkDestroyShaderModule(c.device,sm,NULL);
    VkDescriptorPoolSize ps[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo dpci={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=1; dpci.poolSizeCount=3; dpci.pPoolSizes=ps;
    VK_CHECK(vkCreateDescriptorPool(c.device,&dpci,NULL,&cp.descPool));
    VkDescriptorSetAllocateInfo dsai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=cp.descPool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&cp.descLayout;
    VK_CHECK(vkAllocateDescriptorSets(c.device,&dsai,&cp.descSet));
    return cp;
}

void update_desc(VulkanCtx& c, ComputePipe& cp, VkAccelerationStructureKHR tlas, Buffer& faceBuf, Buffer& mediaBuf, Buffer& outBuf, Buffer& paramBuf, Buffer& seedBuf) {
    VkWriteDescriptorSetAccelerationStructureKHR asW={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR}; asW.accelerationStructureCount=1; asW.pAccelerationStructures=&tlas;
    VkDescriptorBufferInfo fi={faceBuf.buf,0,VK_WHOLE_SIZE}, mi={mediaBuf.buf,0,VK_WHOLE_SIZE}, oi={outBuf.buf,0,VK_WHOLE_SIZE}, pi={paramBuf.buf,0,VK_WHOLE_SIZE}, si={seedBuf.buf,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[6]; memset(w,0,sizeof(w));
    w[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].pNext=&asW; w[0].dstSet=cp.descSet; w[0].dstBinding=0; w[0].descriptorCount=1; w[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    for(int i=1;i<=5;i++){w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=cp.descSet;w[i].dstBinding=(uint32_t)i;w[i].descriptorCount=1;w[i].descriptorType=(i==4)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;}
    w[1].pBufferInfo=&fi; w[2].pBufferInfo=&mi; w[3].pBufferInfo=&oi; w[4].pBufferInfo=&pi; w[5].pBufferInfo=&si;
    vkUpdateDescriptorSets(c.device,6,w,0,NULL);
}

// ---- CLI parsing (matching umcx.cpp) ----
struct CmdOverrides { uint64_t nphoton,batch_size; uint32_t rng_seed,totalthread; float unitinmm; int outputtype,isreflect,isnormalize; int gpuid; bool listgpu; std::string session_id,json_str; bool dumpjson;
    CmdOverrides():nphoton(0),batch_size(0),rng_seed(0),totalthread(0),unitinmm(0),outputtype(-1),isreflect(-1),isnormalize(-1),gpuid(0),listgpu(false),dumpjson(false){} };

void printhelp(const char* n) {
    printf("Vulkan RT-MMC — Ray-tracing accelerated mesh Monte Carlo\nUsage: %s input.json  OR  %s -f input.json [flags]\n\n"
        "Flags:\n -f/--input\tJSON file\n -n/--photon\tphoton number\n -s/--session\tsession name\n -u/--unitinmm\tvoxel size [1]\n"
        " -E/--seed\tRNG seed\n -O/--outputtype\tx:energy,f:flux,l:fluence\n -b/--reflect\tmismatch [1]\n -U/--normalize\t[1]\n"
        " -S/--save2pt\tsave volume [1]\n -t/--thread\tGPU threads [65536]\n -B/--batch\tphotons/batch [500000]\n"
        " -G/--gpuid\tGPU device ID [1]\n -L/--listgpu\tlist all GPUs\n"
        " -j/--json\tJSON override\n --dumpjson\tdump config\n -h/--help\n",n,n);
    exit(0);
}

SimConfig parse_cmdline(int argc, char** argv, CmdOverrides& ovr) {
    std::string inputfile;
    if(argc<2) printhelp(argv[0]);

    for(int i=1;i<argc;i++){
        std::string a(argv[i]);
        if((a=="-f"||a=="--input")&&i+1<argc) inputfile=argv[++i];
        else if((a=="-n"||a=="--photon")&&i+1<argc) ovr.nphoton=(uint64_t)atof(argv[++i]);
        else if((a=="-s"||a=="--session")&&i+1<argc) ovr.session_id=argv[++i];
        else if((a=="-u"||a=="--unitinmm")&&i+1<argc) ovr.unitinmm=(float)atof(argv[++i]);
        else if((a=="-E"||a=="--seed")&&i+1<argc) ovr.rng_seed=(uint32_t)atoi(argv[++i]);
        else if((a=="-O"||a=="--outputtype")&&i+1<argc){char c2=argv[++i][0]; ovr.outputtype=(c2=='f'?0:c2=='l'?1:c2=='x'?2:2);}
        else if((a=="-b"||a=="--reflect")&&i+1<argc) ovr.isreflect=atoi(argv[++i]);
        else if((a=="-U"||a=="--normalize")&&i+1<argc) ovr.isnormalize=atoi(argv[++i]);
        else if((a=="-t"||a=="--thread")&&i+1<argc) ovr.totalthread=(uint32_t)atoi(argv[++i]);
        else if((a=="-B"||a=="--batch")&&i+1<argc) ovr.batch_size=(uint64_t)atof(argv[++i]);
        else if((a=="-G"||a=="--gpuid")&&i+1<argc) ovr.gpuid=atoi(argv[++i]);
        else if(a=="-L"||a=="--listgpu") ovr.listgpu=true;
        else if((a=="-j"||a=="--json")&&i+1<argc) ovr.json_str=argv[++i];
        else if(a=="--dumpjson") ovr.dumpjson=true;
        else if(a=="-h"||a=="--help") printhelp(argv[0]);
        else if(a[0]!='-' && inputfile.empty()) inputfile=a;  // positional arg = input file
    }
    if(inputfile.empty()){fprintf(stderr,"Error: no input JSON file specified\n");printhelp(argv[0]);}
    SimConfig cfg=load_json_input(inputfile.c_str());
    if(ovr.nphoton>0) cfg.nphoton=ovr.nphoton;
    if(ovr.rng_seed>0) cfg.rng_seed=ovr.rng_seed;
    if(!ovr.session_id.empty()) cfg.session_id=ovr.session_id;
    if(ovr.unitinmm>0) cfg.unitinmm=ovr.unitinmm;
    if(ovr.outputtype>=0) cfg.output_type=ovr.outputtype;
    if(ovr.isreflect>=0) cfg.do_mismatch=(ovr.isreflect!=0);
    if(ovr.isnormalize>=0) cfg.do_normalize=(ovr.isnormalize!=0);
    return cfg;
}

// ---- Main ----
int main(int argc, char** argv) {
    const char* spirvFile="vkmmc_core.spv";
    // Check if last arg is .spv
    if(argc>2){std::string la(argv[argc-1]); if(la.size()>4&&la.substr(la.size()-4)==".spv"){spirvFile=argv[--argc];}}

    CmdOverrides ovr;
    // Handle -L before full parse (doesn't need input file)
    for(int i=1;i<argc;i++){std::string a(argv[i]); if(a=="-L"||a=="--listgpu"){list_vulkan_gpus();return 0;}}
    SimConfig cfg=parse_cmdline(argc,argv,ovr);
    if(ovr.dumpjson){printf("{\"Session\":{\"ID\":\"%s\",\"Photons\":%lu},\"Mesh\":{\"Nodes\":%lu,\"Faces\":%lu}}\n",cfg.session_id.c_str(),(unsigned long)cfg.nphoton,cfg.nodes.size(),cfg.faces.size());return 0;}

    VulkanCtx ctx=init_vulkan(ovr.gpuid);

    // ---- Build BLAS (Single-AS) ----
    AccelStruct blas, tlas;
    {
        const VkBufferUsageFlags gu=VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        const VkMemoryPropertyFlags hv=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        Buffer vb=create_buffer(ctx,cfg.nodes.size()*sizeof(Vec3),gu,hv); upload_host(ctx,vb,cfg.nodes.data(),cfg.nodes.size()*sizeof(Vec3));
        std::vector<uint32_t> idx; idx.reserve(cfg.faces.size()*3);
        for(size_t i=0;i<cfg.faces.size();i++){idx.push_back(cfg.faces[i][0]);idx.push_back(cfg.faces[i][1]);idx.push_back(cfg.faces[i][2]);}
        Buffer ib=create_buffer(ctx,idx.size()*sizeof(uint32_t),gu,hv); upload_host(ctx,ib,idx.data(),idx.size()*sizeof(uint32_t));
        VkAccelerationStructureGeometryTrianglesDataKHR tris={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        tris.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT; tris.vertexData.deviceAddress=get_addr(ctx,vb.buf); tris.vertexStride=sizeof(Vec3);
        tris.maxVertex=(uint32_t)cfg.nodes.size()-1; tris.indexType=VK_INDEX_TYPE_UINT32; tris.indexData.deviceAddress=get_addr(ctx,ib.buf);
        VkAccelerationStructureGeometryKHR geom={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geom.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geom.geometry.triangles=tris; geom.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureBuildGeometryInfoKHR bi={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bi.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR; bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; bi.geometryCount=1; bi.pGeometries=&geom;
        uint32_t pc=(uint32_t)cfg.faces.size(); VkAccelerationStructureBuildSizesInfoKHR sz={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        ctx.pfnGetASBuildSizes(ctx.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&bi,&pc,&sz);
        blas.buffer=create_buffer(ctx,sz.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR asci={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; asci.buffer=blas.buffer.buf; asci.size=sz.accelerationStructureSize; asci.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        ctx.pfnCreateAS(ctx.device,&asci,NULL,&blas.handle);
        Buffer scratch=create_buffer(ctx,sz.buildScratchSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        bi.dstAccelerationStructure=blas.handle; bi.scratchData.deviceAddress=get_addr(ctx,scratch.buf);
        VkAccelerationStructureBuildRangeInfoKHR range; memset(&range,0,sizeof(range)); range.primitiveCount=pc;
        const VkAccelerationStructureBuildRangeInfoKHR* pR=&range;
        VkCommandBuffer cmd=begin_cmd(ctx); ctx.pfnCmdBuildAS(cmd,1,&bi,&pR); end_submit(ctx,cmd);
        destroy_buf(ctx,scratch); destroy_buf(ctx,vb); destroy_buf(ctx,ib);
        printf("BLAS: %u triangles, %lu KB\n",pc,(unsigned long)(sz.accelerationStructureSize/1024));
        tlas=build_tlas(ctx,blas);
    }

    const VkBufferUsageFlags ssbo=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // GPU buffers (device-local)
    Buffer faceBuf=create_device_buffer(ctx,cfg.facedata.size()*sizeof(FaceData),ssbo);
    upload_to_device(ctx,faceBuf,cfg.facedata.data(),cfg.facedata.size()*sizeof(FaceData));
    Buffer mediaBuf=create_device_buffer(ctx,cfg.media.size()*sizeof(Medium),ssbo);
    upload_to_device(ctx,mediaBuf,cfg.media.data(),cfg.media.size()*sizeof(Medium));

    // Output grid
    float voxel_size=cfg.unitinmm; if(cfg.has_steps) voxel_size=cfg.steps[0];
    float grid_eps=voxel_size*0.5f;
    float gmin[3]={cfg.nmin.x-grid_eps,cfg.nmin.y-grid_eps,cfg.nmin.z-grid_eps};
    float gmax[3]={cfg.nmax.x+grid_eps,cfg.nmax.y+grid_eps,cfg.nmax.z+grid_eps};
    uint32_t nx=(uint32_t)ceil((gmax[0]-gmin[0])/voxel_size);
    uint32_t ny=(uint32_t)ceil((gmax[1]-gmin[1])/voxel_size);
    uint32_t nz=(uint32_t)ceil((gmax[2]-gmin[2])/voxel_size);
    uint32_t crop0w=nx*ny*nz*cfg.maxgate, outSize=crop0w*2;

    Buffer outBuf=create_device_buffer(ctx,outSize*sizeof(float),ssbo);
    zero_device_buffer(ctx,outBuf,outSize*sizeof(float));

    // Thread/batch setup
    uint32_t totalthread=(ovr.totalthread>0)?ovr.totalthread:65536;
    if(cfg.nphoton<totalthread){totalthread=((uint32_t)cfg.nphoton+255)/256*256; if(!totalthread)totalthread=256;}

    printf("Grid: %ux%ux%u x %d gates, voxel=%.3fmm, origin=[%.2f,%.2f,%.2f]\n",nx,ny,nz,cfg.maxgate,voxel_size,gmin[0],gmin[1],gmin[2]);

    MCParams params; memset(&params,0,sizeof(params));
    params.srctype=cfg.srctype;
    for(int i=0;i<3;i++){params.srcpos[i]=cfg.srcpos[i];params.srcdir[i]=cfg.srcdir[i];}
    for(int i=0;i<4;i++){params.srcparam1[i]=cfg.srcparam1[i];params.srcparam2[i]=cfg.srcparam2[i];}
    params.nmin[0]=gmin[0];params.nmin[1]=gmin[1];params.nmin[2]=gmin[2];
    params.nmax[0]=gmax[0]-gmin[0];params.nmax[1]=gmax[1]-gmin[1];params.nmax[2]=gmax[2]-gmin[2];
    params.crop0[0]=nx; params.crop0[1]=nx*ny; params.crop0[2]=nx*ny*nz; params.crop0[3]=crop0w;
    params.dstep=1.0f/voxel_size; params.tstart=cfg.t0; params.tend=cfg.t1; params.Rtstep=1.0f/cfg.dt;
    params.maxgate=cfg.maxgate; params.mediumid0=cfg.mediumid0;
    params.isreflect=cfg.do_mismatch?1u:0u; params.outputtype=cfg.output_type;
    params.total_threads=totalthread; params.num_media=(uint32_t)cfg.media.size(); params.seed=cfg.rng_seed;

    Buffer paramBuf=create_device_buffer(ctx,sizeof(MCParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Seeds
    srand(cfg.rng_seed>0?cfg.rng_seed:(uint32_t)time(0));
    struct uint4_t{uint32_t x,y,z,w;};
    std::vector<uint4_t> seeds(totalthread);
    for(uint32_t i=0;i<totalthread;i++) seeds[i]={(uint32_t)rand(),(uint32_t)rand(),(uint32_t)rand(),(uint32_t)rand()};
    Buffer seedBuf=create_device_buffer(ctx,totalthread*sizeof(uint4_t),ssbo);
    upload_to_device(ctx,seedBuf,seeds.data(),totalthread*sizeof(uint4_t));

    ComputePipe cp=create_pipeline(ctx,spirvFile);
    update_desc(ctx,cp,tlas.handle,faceBuf,mediaBuf,outBuf,paramBuf,seedBuf);

    // Batched dispatch
    uint64_t photons_per_batch=(ovr.batch_size>0)?ovr.batch_size:500000;
    uint64_t photons_done=0; int batch=0;
    uint32_t wg=totalthread/256;
    printf("Threads: %u (%u workgroups), batch: %lu photons\n",totalthread,wg,(unsigned long)photons_per_batch);

    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point t0=Clock::now();

    while(photons_done<cfg.nphoton){
        uint64_t rem=cfg.nphoton-photons_done;
        uint64_t bp=std::min(rem,photons_per_batch);
        params.threadphoton=(int)(bp/totalthread);
        params.oddphoton=(int)(bp-(uint64_t)params.threadphoton*totalthread);
        upload_to_device(ctx,paramBuf,&params,sizeof(params));

        VkCommandBuffer cmd=begin_cmd(ctx);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,cp.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,cp.pipeLayout,0,1,&cp.descSet,0,NULL);
        vkCmdDispatch(cmd,wg,1,1);
        end_submit(ctx,cmd);

        photons_done+=bp; batch++;
        printf("  batch %d: %lu photons (%lu/%lu)\n",batch,(unsigned long)bp,(unsigned long)photons_done,(unsigned long)cfg.nphoton);
    }

    double ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    printf("Complete (%d batches), speed: %.2f photon/ms, duration: %.3f ms\n",batch,(double)cfg.nphoton/ms,ms);

    // Readback
    std::vector<float> raw(outSize); download_from_device(ctx,outBuf,raw.data(),outSize*sizeof(float));
    std::vector<float> fluence(crop0w);
    for(uint32_t i=0;i<crop0w;i++) fluence[i]=raw[i]+raw[i+crop0w];
    if(cfg.do_normalize){float vv=voxel_size*voxel_size*voxel_size; for(uint32_t i=0;i<crop0w;i++) fluence[i]/=(float)cfg.nphoton*vv;}

    // Save JData JSON
    std::string outname=cfg.session_id+".jdat";
    {
        std::vector<size_t> dims; if(cfg.maxgate>1) dims={nx,ny,nz,(size_t)cfg.maxgate}; else dims={nx,ny,nz};
        json root;
        root["Session"]={{"ID",cfg.session_id},{"Photons",cfg.nphoton}};
        root["Forward"]={{"T0",cfg.t0},{"T1",cfg.t1},{"Dt",cfg.dt}};
        root["Domain"]={{"LengthUnit",cfg.unitinmm},{"VoxelSize",voxel_size},{"Dim",{nx,ny,nz}},{"Origin",{gmin[0],gmin[1],gmin[2]}}};
        root["Fluence"]=jdata_encode("single",dims,fluence.data(),crop0w*sizeof(float));
        std::ofstream f(outname); f<<root.dump(2)<<std::endl;
        printf("Output: %s (%ux%ux%u",outname.c_str(),nx,ny,nz); if(cfg.maxgate>1)printf("x%d",cfg.maxgate); printf(")\n");
    }

    // Cleanup
    ctx.pfnDestroyAS(ctx.device,tlas.handle,NULL); ctx.pfnDestroyAS(ctx.device,blas.handle,NULL);
    destroy_buf(ctx,tlas.buffer); destroy_buf(ctx,blas.buffer);
    destroy_buf(ctx,faceBuf); destroy_buf(ctx,mediaBuf); destroy_buf(ctx,outBuf); destroy_buf(ctx,paramBuf); destroy_buf(ctx,seedBuf);
    vkDestroyPipeline(ctx.device,cp.pipeline,NULL); vkDestroyPipelineLayout(ctx.device,cp.pipeLayout,NULL);
    vkDestroyDescriptorPool(ctx.device,cp.descPool,NULL); vkDestroyDescriptorSetLayout(ctx.device,cp.descLayout,NULL);
    vkDestroyCommandPool(ctx.device,ctx.cmdPool,NULL); vkDestroyDevice(ctx.device,NULL); vkDestroyInstance(ctx.instance,NULL);
    return 0;
}