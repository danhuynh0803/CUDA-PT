OBJECTS = cuda_pt.exe cuda_pt.exp cuda_pt.lib 

CC = nvcc
CFLAGS = -g 

ifdef ComSpec
	RM = del /F /Q
else 
	RM = rm -f
endif 

pt: 
	$(CC) $(CFLAGS) cuda_pt.cu -o cuda_pt 

clean: 
	$(RM) $(OBJECTS)
