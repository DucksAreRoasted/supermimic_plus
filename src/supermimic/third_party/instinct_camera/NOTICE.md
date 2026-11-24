Camera implementation adapted from project-instinct/InstinctMJ at
03563ee02665f327425106372264070315673748, distributed under the upstream
CC BY-NC 4.0 license (see LICENSE). Source: https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj .
The grouped sensor and private-world ray kernel are retained. Imports are
local; the depth pipeline is specialized to the G1 configuration and accepts
raw image tensors independently of the upstream task managers.
