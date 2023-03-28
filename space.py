import torch



print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())

###del model
torch.cuda.empty_cache()
