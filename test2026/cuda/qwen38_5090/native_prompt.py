"""Pure prompt positioning shared by the native study worker and smoke probe."""
def image_positions(count,image_start,image_tokens,merged_width):
    if count<1 or image_start<0 or image_tokens<1 or merged_width<1 or image_tokens%merged_width or image_start+image_tokens>count:
        raise ValueError('Invalid merged image token geometry')
    current=0;positions=[]
    for i in range(count):
        if image_start<=i<image_start+image_tokens:
            j=i-image_start;positions.append([current,current+j//merged_width,current+j%merged_width])
            if j+1==image_tokens:current+=max(image_tokens//merged_width,merged_width)
        else:positions.append([current]*3);current+=1
    return positions,current
