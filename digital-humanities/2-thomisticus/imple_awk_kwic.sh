#http://corpus.wordish.org/hacks/hack-20-diy-kwic-with-awk
#Another way qould be to create a bash script containing the following
#It saves to hardcode the pattern to look for
#!/bin/bash
read -e -p 'Search (regex): ' _SEARCH
read -p '# left: ' _LEFT
read -p '# right: ' _RIGHT
awk -v _left=${_LEFT} -v _right=${_RIGHT} -v _search="${_SEARCH}" '
match($0,_search){
print $1": "substr($0,RSTART-_left,RLENGTH+_left+_right)
}
' OpenLibrary-CorpusThomisticus.txt \
| egrep -i "${_SEARCH}" | less
#you can run the script by invoking
#./simple_awk_kwic.sh