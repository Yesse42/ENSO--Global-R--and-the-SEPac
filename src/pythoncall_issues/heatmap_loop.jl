cd(@__DIR__)

using PythonCall

@py import matplotlib.pyplot as plt

i = 0
while true
    global i
    i += 1

    fig, ax = plt.subplots(1, 1)
    plt.close(fig)
    PythonCall.pydel!(ax)
    PythonCall.pydel!(fig)
    GC.gc()

    println("Iteration $i done")
end
