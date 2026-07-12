#pragma once

namespace Garnet
{
    bool IsCompiledGraphCaptureActive();
    bool IsCompiledFusionCaptureRootActive();

    class ScopedCompiledGraphCapture
    {
        bool m_previous = false;

    public:
        ScopedCompiledGraphCapture();
        ~ScopedCompiledGraphCapture();

        ScopedCompiledGraphCapture(const ScopedCompiledGraphCapture&) = delete;
        ScopedCompiledGraphCapture& operator=(const ScopedCompiledGraphCapture&) = delete;
    };

    class ScopedCompiledFusionCaptureRoot
    {
    public:
        ScopedCompiledFusionCaptureRoot();
        ~ScopedCompiledFusionCaptureRoot();

        ScopedCompiledFusionCaptureRoot(const ScopedCompiledFusionCaptureRoot&) = delete;
        ScopedCompiledFusionCaptureRoot& operator=(const ScopedCompiledFusionCaptureRoot&) = delete;
    };
}
