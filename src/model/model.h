#pragma once

#include "xpackage.h"
#include "xlang.h"

namespace Garnet
{
	class Model
	{
		X::Value mModel;//a dictionary to store {key:tersor}
	public:
		BEGIN_PACKAGE(Model)
			END_PACKAGE
	};
}