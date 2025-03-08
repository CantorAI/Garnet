#include "model.h"
namespace Garnet
{
    X::Value Model::Access(X::Port::vector<X::Value>& IdxAry)
    {
        for (auto& idx : IdxAry)
        {
            if (idx.IsLong())
            {
                int nIdx = (int)idx.GetLongLong();
            }
            else
            {
                std::string strIdx = idx.ToString();
                return mModel[strIdx];
            }
        }
        return X::Value();
    }
}