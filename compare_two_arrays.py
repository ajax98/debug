import numpy as np
import argparse

def highlight_differences_with_margin_4d(array1, array2, relative_margin=0.1):
  """
  Compares two NumPy arrays and highlights elements that differ by more than
  a specified relative margin.

  Args:
    array1: The first NumPy array.
    array2: The second NumPy array.
    relative_margin: The relative margin for comparison (e.g., 0.1 for 10%).

  Returns:
    A formatted string highlighting the differences.
  """

  if array1.shape != array2.shape:
    return "Arrays have different shapes and cannot be compared."

  # Calculate the absolute difference between the arrays
  abs_diff = np.abs(array1 - array2)

  # Calculate the relative difference
  denom = np.maximum(np.abs(array1), np.abs(array2)) + 1e-20
  relative_diff = abs_diff / denom

  # Find indices where the relative difference exceeds the margin
  indices = np.where(relative_diff > relative_margin)

  highlighted = ""
  for i in range(len(indices[0])):
    x = indices[0][i]
    y = indices[1][i]
    z = indices[2][i]
    a = indices[3][i]
    highlighted += (f"Difference at index ({x}, {y}, {z}, {a}): " +
        f"{array1[x, y, z, a]} != {array2[x, y, z, a]} " +
        f"(relative diff: {relative_diff[x, y, z, a]:.2%})\n")
    

  if not highlighted:
    return "No differences exceeding the relative margin."
  else:
    return highlighted

def highlight_differences_with_margin(array1, array2, relative_margin=0.1):
  """
  Compares two NumPy arrays and highlights elements that differ by more than
  a specified relative margin.

  Args:
    array1: The first NumPy array.
    array2: The second NumPy array.
    relative_margin: The relative margin for comparison (e.g., 0.1 for 10%).

  Returns:
    A formatted string highlighting the differences.
  """

  if array1.shape != array2.shape:
    return "Arrays have different shapes and cannot be compared."

  # Calculate the absolute difference between the arrays
  abs_diff = np.abs(array1 - array2)

  # Calculate the relative difference
  denom = np.maximum(np.abs(array1), np.abs(array2)) + 1e-20
  relative_diff = abs_diff / denom

  # Find indices where the relative difference exceeds the margin
  indices = np.where(relative_diff > relative_margin)

  highlighted = ""
  for i in range(len(indices[0])):
    x = indices[0][i]
    y = indices[1][i]
    z = indices[2][i]
    highlighted += (f"Difference at index ({x}, {y}, {z}): " +
        f"{array1[x, y, z]} != {array2[x, y, z]} " +
        f"(relative diff: {relative_diff[x, y, z]:.2%})\n")
    

  if not highlighted:
    return "No differences exceeding the relative margin."
  else:
    return highlighted

def main():
  parser = argparse.ArgumentParser(description="Compare two arrays")

  # Add arguments
  parser.add_argument("arr_1_path", type=str, help="array 1 path")
  parser.add_argument("arr_2_path", type=str, help="array 2 path")

  args = parser.parse_args()

  s = np.load(args.arr_1_path)
  f = np.load(args.arr_2_path)

  print(highlight_differences_with_margin_4d(s, f))

if __name__ == "__main__":
  main()