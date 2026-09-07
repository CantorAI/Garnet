#pragma once
#include "xlang3/xlang3.h"
#include <limits>
#include <type_traits>

namespace Garnet {
void ValidateDenseTensor(const X::Tensor& tensor, bool writing = false);

inline int CheckedInt(int64_t value, const char* name) {
    if (value < INT32_MIN || value > INT32_MAX) throw X::Error(std::string(name) + " exceeds int32 range");
    return static_cast<int>(value);
}
inline int64_t CheckedInt64(const X::Value& value, const char* name) {
    if ((!value.IsInt64() && !value.IsUInt64()) ||
        (value.IsUInt64() && value.ToUInt64() > INT64_MAX))
        throw X::Error(std::string(name) + " requires an int64 value");
    return value.ToLongLong();
}
inline int CheckedInt(const X::Value& value, const char* name) {
    return CheckedInt(CheckedInt64(value, name), name);
}

inline uint64_t TensorCount(const X::Tensor& tensor) {
    const auto info = tensor.Info();
    if (info.rank == UINT32_MAX) throw X::Error("tensor shape is not inferred");
    uint64_t count = 1;
    for (uint32_t i = 0; i < info.rank; ++i) {
        if (info.shape[i] < 0 || (info.shape[i] && count > UINT64_MAX / info.shape[i]))
            throw X::Error("tensor element count is out of range");
        count *= info.shape[i];
    }
    return count;
}

inline X::Value FindField(const X::Value& value, const char* name) {
    if (!value.IsValid()) return {};
    if (!value.IsDict()) {
        X::Module builtins(value.host(), "builtins");
        auto getter = builtins["getattr"];
        X::Value result;
        if (!getter.Call({value, X::Value::String(value.host(), name), X::Value(nullptr)}, result))
            throw X::Error(value.host()->runtime_last_error(value.runtime()));
        return result.IsNone() ? X::Value() : result;
    }
    for (uint64_t i = 0; i < value.Size(); ++i) {
        X::Value key, item;
        if (!value.DictEntry(i, key, item)) throw X::Error("cannot read dictionary entry");
        if (key.IsString() && key.ToString() == name) return item;
    }
    return {};
}

inline X::Value NativeValue(X3PackageHost*, const X::Value& value) { return value; }
inline X::Value NativeValue(X3PackageHost*, X::Value&& value) { return std::move(value); }
inline X::Value NativeValue(X3PackageHost* host, const std::string& value) { return X::Value::String(host, value); }
inline X::Value NativeValue(X3PackageHost* host, const char* value) { return X::Value::String(host, value); }
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
X::Value NativeValue(X3PackageHost*, T value) { return X::Value(value); }

template<class... Args>
X::Value CallChecked(const X::Value& callable, Args&&... args) {
    X::ARGS values{NativeValue(callable.host(), std::forward<Args>(args))...};
    X::Value result;
    if (!callable.Call(values, result))
        throw X::Error(callable.host() ? callable.host()->runtime_last_error(callable.runtime()) : "missing callable");
    return result;
}
}
