#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt


# sed "s/^\s\+//g" = strip of beginning whitespaces and '+' signs for some reason
# sort -rn = sort based on initial number (n) and reverse (r)
# grep -v "^[1234]\s" = match all non-occurences of strings beginning with 1,2,3 or 4 and a whitespace
# cut -d' ' -f2 = split lines into columns delimited by ' ' and cut out second column
