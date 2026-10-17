#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(){@autoreleasepool{
if(@available(macOS 15.0,*)){
 auto d=MTLCreateSystemDefaultDevice();if(!d)return 2;
 auto desc=[MTLResidencySetDescriptor new];desc.initialCapacity=4;desc.label=@"Garnet bounded feasibility";NSError* error=nil;
 auto set=[d newResidencySetWithDescriptor:desc error:&error];if(!set){fprintf(stderr,"%s\n",error.description.UTF8String);return 3;}
 const NSUInteger bytes=1024*1024;auto b=[d newBufferWithLength:bytes options:MTLResourceStorageModeShared];auto q=[d newCommandQueue];if(!b||!q)return 4;
 [set addAllocation:b];[set commit];if(set.allocationCount!=1||![set containsAllocation:b])return 5;
 [q addResidencySet:set];[set requestResidency];auto c=[q commandBuffer];auto encoder=[c blitCommandEncoder];[encoder fillBuffer:b range:NSMakeRange(0,bytes) value:0x5a];[encoder endEncoding];[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted)return 6;
 for(NSUInteger i=0;i<bytes;i++)if(((const unsigned char*)b.contents)[i]!=0x5a)return 7;
 const auto allocationBytes=set.allocatedSize;[q removeResidencySet:set];[set endResidency];[set removeAllocation:b];[set commit];if(set.allocationCount!=0||[set containsAllocation:b])return 8;
 printf("{\"device\":\"%s\",\"buffer_bytes\":%lu,\"set_allocated_size\":%llu,\"queue_attach_request_and_blit_exact\":true,\"detach_end_remove_commit\":true,\"scope\":\"Basic1MiBAPI lifecycle only;allocatedSize is not proof of physical residency, no model correctness or performance evidence\"}\n",d.name.UTF8String,(unsigned long)bytes,(unsigned long long)allocationBytes);return 0;
}else{return 9;}
}}
