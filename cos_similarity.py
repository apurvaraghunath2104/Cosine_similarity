import numpy as np
from utils import *



# Function to get cosine similarity 
# Arguments:
# a: A numpy vector of size (x, )
# b: A numpy vector of size (x, )
# Returns: sim (float)
# Where, sim (float) is the cosine similarity between vectors a and b. x is the size of the numpy vector.
def cosine_similarity(a, b):
	sum_aa = 0
	sum_ab = 0
	sum_bb = 0
	for i in range(0,len(a)):
		x = a[i]
		y = b[i]
		sum_aa += x**2
		sum_bb += y**2
		sum_ab += x*y
	res = sum_ab/((sum_aa**0.5)*(sum_bb**0.5))
	return res



def main():
	# Initialize the vectors
	a = np.array([1, 2, 3])
	b = np.array([7, 8, 9])

	# We can print the vectors as
	print("Vector a: {0}".format(a))
	print("Vector b: {0}".format(b))

	# We can see shape (and dimensions) of vectors as
	print("Vector a shape: {0}".format(a.shape))
	print("Vector b shape: {0}".format(b.shape))

	# Compute cosine similarity
	sim = cosine_similarity(a, b)
	print("Cosine similarity between inputs: {0:.4}".format(sim))

	# We can load numpy word vectors using load_w2v as
	w2v = load_w2v()
	# And access these vectors using the dictionary
	word = w2v['quarantine']
	dist1 = w2v['distancing']
	dist2 = w2v['pandemic']
	dist3 = w2v['party']
	dist4 = w2v['sourdough']
	sim1 = cosine_similarity(word, dist1)
	print("Cosine similarity between quarantine & distancing: {0:.4}".format(sim1))

	sim2 = cosine_similarity(word, dist2)
	print("Cosine similarity between quarantine & pandemic: {0:.4}".format(sim2))

	sim3 = cosine_similarity(word, dist3)
	print("Cosine similarity between quarantine & party: {0:.4}".format(sim3))

	sim4 = cosine_similarity(word, dist4)
	print("Cosine similarity between quarantine & sourdough: {0:.4}".format(sim4))


if __name__ == '__main__':
	main()
