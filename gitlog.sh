#!/bin/sh
# print the commit history in the ChangeLog.txt format

git log --pretty=" %ad [%h] %s" --date=short
