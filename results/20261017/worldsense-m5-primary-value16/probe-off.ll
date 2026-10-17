; ModuleID = '/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-value16/probe-off.metal'
source_filename = "/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-value16/probe-off.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
define void @value16_probe(i32 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %0, <4 x i16> addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = mul i32 %2, 3
  %5 = zext i32 %4 to i64
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %5
  %7 = bitcast i32 addrspace(1)* %6 to <4 x i8> addrspace(1)*
  %8 = load <4 x i8>, <4 x i8> addrspace(1)* %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = add i32 %4, 1
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %10
  %12 = load i32, i32 addrspace(1)* %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = add i32 %4, 2
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %14
  %16 = load i32, i32 addrspace(1)* %15, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %17 = tail call <4 x i32> @air.convert.u.v4i32.u.v4i8(<4 x i8> %8) #2
  %18 = and i32 %12, 65535
  %19 = insertelement <4 x i32> undef, i32 %18, i64 0
  %20 = lshr i32 %18, 4
  %21 = insertelement <4 x i32> %19, i32 %20, i64 1
  %22 = lshr i32 %18, 8
  %23 = insertelement <4 x i32> %21, i32 %22, i64 2
  %24 = lshr i32 %18, 12
  %25 = insertelement <4 x i32> %23, i32 %24, i64 3
  %26 = and <4 x i32> %25, <i32 15, i32 15, i32 15, i32 15>
  %27 = and <4 x i32> %17, <i32 127, i32 127, i32 127, i32 127>
  %28 = shl <4 x i32> %17, <i32 8, i32 8, i32 8, i32 8>
  %29 = and <4 x i32> %28, <i32 32768, i32 32768, i32 32768, i32 32768>
  %30 = insertelement <4 x i32> poison, i32 %16, i64 0
  %31 = shufflevector <4 x i32> %30, <4 x i32> poison, <4 x i32> zeroinitializer
  %32 = add <4 x i32> %26, %31
  %33 = shl <4 x i32> %32, <i32 7, i32 7, i32 7, i32 7>
  %34 = or <4 x i32> %33, %27
  %35 = or <4 x i32> %34, %29
  %36 = tail call <4 x i16> @air.convert.u.v4i16.u.v4i32(<4 x i32> %35) #2
  %37 = zext i32 %2 to i64
  %38 = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %1, i64 %37
  store <4 x i16> %36, <4 x i16> addrspace(1)* %38, align 8, !tbaa !31, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare <4 x i32> @air.convert.u.v4i32.u.v4i8(<4 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare <4 x i16> @air.convert.u.v4i16.u.v4i32(<4 x i32>) local_unnamed_addr #1

attributes #0 = { argmemonly mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (i32 addrspace(1)*, <4 x i16> addrspace(1)*, i32)* @value16_probe, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"output"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-value16/probe-off.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(value16_probe)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = !{!24, !24, i64 0}
