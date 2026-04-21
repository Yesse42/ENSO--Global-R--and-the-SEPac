cd(@__DIR__)

using PythonCall

@py import numpy as np

i = 0
while true
    global i
    i += 1

    arr = np.random.rand(1000, 1000)
    PythonCall.pydel!(arr)
    GC.gc()

    println("Iteration $i done")
end
