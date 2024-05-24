#%%
def isPairSum(A, N, X):
 
    for i in range(N):
        for j in range(N):
 
            # as equal i and j means same element
            if(i == j):
                continue
 
            # pair exists
            if (A[i] + A[j] == X):
                return True
 
            # as the array is sorted
            if (A[i] + A[j] > X):
                break
             
    # No pair found with given sum
    return 0
 
# Driver code
arr = [3, 5, 9, 2, 8, 10, 11]
val = 17
 
print(isPairSum(arr, len(arr), val))
# %%
