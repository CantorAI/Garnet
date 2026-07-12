#include "compiled_graph_capture.h"

namespace Garnet
{
    namespace
    {
        thread_local bool g_compiledGraphCaptureActive = false;
        thread_local int g_compiledFusionCaptureDepth = 0;
    }

    bool IsCompiledFusionCaptureRootActive()
    {
        return g_compiledFusionCaptureDepth > 0;
    }

    bool IsCompiledGraphCaptureActive()
    {
        return g_compiledGraphCaptureActive;
    }

    ScopedCompiledGraphCapture::ScopedCompiledGraphCapture()
        : m_previous(g_compiledGraphCaptureActive)
    {
        g_compiledGraphCaptureActive = true;
    }

    ScopedCompiledGraphCapture::~ScopedCompiledGraphCapture()
    {
        g_compiledGraphCaptureActive = m_previous;
    }

    ScopedCompiledFusionCaptureRoot::ScopedCompiledFusionCaptureRoot()
    {
        ++g_compiledFusionCaptureDepth;
    }

    ScopedCompiledFusionCaptureRoot::~ScopedCompiledFusionCaptureRoot()
    {
        --g_compiledFusionCaptureDepth;
    }
}
