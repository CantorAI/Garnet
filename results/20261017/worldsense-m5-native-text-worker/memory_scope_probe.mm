#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "nlohmann/json.hpp"
extern char **environ;
using json=nlohmann::json;
static void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
static uint64_t footprint(){task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;require(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS,"task_info failed");return info.phys_footprint;}
struct OwnedChild{pid_t pid=0;int input=-1,output=-1;~OwnedChild(){if(input>=0)close(input);if(output>=0)close(output);if(pid>0){int status=0;for(int i=0;i<100;++i){auto r=waitpid(pid,&status,WNOHANG);if(r==pid||r==-1){pid=0;break;}usleep(10000);}if(pid>0){kill(pid,SIGTERM);waitpid(pid,&status,0);}}}};
int main(int argc,char**argv){@autoreleasepool{try{
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"NoMetaldevice");
 if(argc==2){
  uint64_t before=device.currentAllocatedSize;id<MTLBuffer> buffer=[device newBufferWithLength:32*1024*1024 options:MTLResourceStorageModeShared];require(buffer!=nil,"child buffer allocation failed");std::memset(buffer.contents,0x5a,buffer.length);
  auto allocated=buffer.allocatedSize;std::printf("%llu %llu %llu %llu\n",(unsigned long long)before,(unsigned long long)device.currentAllocatedSize,(unsigned long long)allocated,(unsigned long long)footprint());std::fflush(stdout);
  char done=0;require(read(STDIN_FILENO,&done,1)==1,"parent closed before release");buffer=nil;return 0;
 }
 uint64_t before=device.currentAllocatedSize;id<MTLBuffer> own=[device newBufferWithLength:16*1024*1024 options:MTLResourceStorageModeShared];require(own!=nil,"parent allocation failed");std::memset(own.contents,0x36,own.length);uint64_t held=device.currentAllocatedSize;require(held>=before+own.allocatedSize,"parent allocation not reflected");
 int input[2],output[2];require(pipe(input)==0&&pipe(output)==0,"pipe failed");posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);posix_spawn_file_actions_adddup2(&actions,input[0],STDIN_FILENO);posix_spawn_file_actions_adddup2(&actions,output[1],STDOUT_FILENO);for(int fd:{input[0],input[1],output[0],output[1]})posix_spawn_file_actions_addclose(&actions,fd);
 OwnedChild child;char mode[]="--child";char*args[]={argv[0],mode,nullptr};int result=posix_spawn(&child.pid,argv[0],&actions,nullptr,args,environ);posix_spawn_file_actions_destroy(&actions);close(input[0]);close(output[1]);child.input=input[1];child.output=output[0];require(result==0,"spawn failed");
 FILE* stream=fdopen(dup(child.output),"r");require(stream!=nullptr,"fdopen failed");unsigned long long cb=0,ca=0,allocation=0,cf=0;int fields=fscanf(stream,"%llu %llu %llu %llu",&cb,&ca,&allocation,&cf);fclose(stream);require(fields==4,"child snapshot malformed");uint64_t parentWithChild=device.currentAllocatedSize;auto pf=footprint();
 require(allocation>=32*1024*1024&&ca>=cb+allocation,"child allocation not reflected");require(cb<16*1024*1024,"child sees parent allocation; scope unproven");require(parentWithChild==held,"parent includes child allocation; cannot sum counters");require(write(child.input,"x",1)==1,"release failed");close(child.input);child.input=-1;int status=0;require(waitpid(child.pid,&status,0)==child.pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"child exit failed");child.pid=0;
 json report={{"parent_pid",getpid()},{"parent_before_bytes",before},{"parent_own_allocated_bytes",uint64_t(own.allocatedSize)},{"parent_held_bytes",held},{"parent_while_child_holds_bytes",parentWithChild},{"child_before_bytes",cb},{"child_held_bytes",ca},{"child_own_allocated_bytes",allocation},{"parent_task_physical_footprint_bytes",pf},{"child_task_physical_footprint_bytes",cf},{"conservative_sum_task_footprint_bytes",pf+cf},{"Metal_allocations_process_local_observed",true},{"no_inference_or_physical_residency_performance_claim",true}};std::cout<<report.dump(2)<<'\n';own=nil;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
