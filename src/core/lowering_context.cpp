#include "lowering_context.h"

namespace Garnet
{
    thread_local ILoweringContext* g_loweringContext = nullptr;

    ScopedLoweringContext::ScopedLoweringContext(ILoweringContext& context)
        : m_previous(g_loweringContext)
    {
        g_loweringContext = &context;
    }

    ScopedLoweringContext::~ScopedLoweringContext()
    {
        g_loweringContext = m_previous;
    }
}
