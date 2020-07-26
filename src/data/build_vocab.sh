#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat dataset/train_pos.txt dataset/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > dataset/vocab.txt


# cat = concatenate and print
# sed = stream operator
# sed "s/ /\n/g" = substitute ' ' with '\n' for all matches (g)
# grep = search for specific text
# grep -v "^\s*$" = for all non matching lines (-v), where matched is: beginning of sentence (^)
# white space (\s) with 0 or more occurences of it (*) and an end of string
# sort = sort text
# uniq -c = take unique occurences and count total occurences of each !line!
