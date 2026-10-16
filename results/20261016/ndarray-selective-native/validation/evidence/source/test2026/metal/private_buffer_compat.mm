#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <initializer_list>
int main(){@autoreleasepool{
 auto device=MTLCreateSystemDefaultDevice();
 for(NSUInteger bytes:{NSUInteger(1408),NSUInteger(8192),NSUInteger(1048576)}){
  auto a=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  auto c=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  std::printf("{\"bytes\":%lu,\"staging\":%s,\"private\":%s,\"readback\":%s,\"actual_mode\":%lu,\"contents_nonnull\":%s}\n",(unsigned long)bytes,a?"true":"false",b?"true":"false",c?"true":"false",(unsigned long)b.storageMode,b.contents?"true":"false");
 }
 return 0;
}}
