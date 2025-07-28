
#!/bin/bash

# Sequential version (default)
echo "Running chunks sequentially..."
for i in {0..99}
do
    echo "Processing chunk $i"
    python process_chunks.py $i
done

echo "All chunks processed!"