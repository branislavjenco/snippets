def is_anagram(a, b):

  # Immediately disregard if lengths are different
  if len(a) != len(b): return False

  # Make the map
  alphabet_map = {letter: 0 for letter in list("abcdefghijklmnopqrstuvwxyz")}

  # Add for every occurence in string a, subtract for every occurence in string b
  for i in range(0, len(a)):
    alphabet_map[a[i]] = alphabet_map[a[i]] + 1
    alphabet_map[b[i]] = alphabet_map[b[i]] - 1

  # Return False if there are any non-zero values left, True otherwise
  return not any(alphabet_map.values())

print("\nWelcome to Anagram Checker v1.0. Brought to you by North Central Positronics.")
a = input("Type the first string: ")
b = input("Type the second string: ")
print("Comparing strings '" + a + "' and '" + b + "'")
if is_anagram(a,b): print("Strings '" + a + "' and '" + b + "' are anagrams!")
else: print("Strings '" + a + "' and '" + b + "' are NOT anagrams!")

