CFLAGS=-g -Wall
CC=gcc

all:sendarp catcharp

sendarp: sendarp.o
	$(CC) $(CFLAGS) sendarp.o -o ipsearch
catcharp: catcharp.o
	$(CC) $(CFLAGS) catcharp.o -o catcharp
	
clean:
	rm -f *.o *.a sendarp catcharp
