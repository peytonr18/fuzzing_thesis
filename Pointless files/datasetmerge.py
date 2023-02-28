import json
from os import path 
import textwrap 
filename = '/Users/peytonrobertson/fuzzing_thesis/function.json'

# Read JSON file
with open(filename) as fp:
    listObj = json.load(fp)

# Verify existing list
#print(listObj)
#print(type(listObj))

sample_func = '''static int uvesafb_setcmap(struct fb_cmap *cmap, struct fb_info *info)
{
  struct uvesafb_pal_entry *entries;
  int shift = 16 - dac_width;
  int i, err = 0;
  if (info->var.bits_per_pixel == 8) {
    if (cmap->start + cmap->len > info->cmap.start +
        info->cmap.len || cmap->start < info->cmap.start)
      return -EINVAL;

    entries = kmalloc_array(cmap->len, sizeof(*entries),
          GFP_KERNEL);
    if (!entries)
      return -ENOMEM;

    for (i = 0; i < cmap->len; i++) {
      entries[i].red   = cmap->red[i]   >> shift;
      entries[i].green = cmap->green[i] >> shift;
      entries[i].blue  = cmap->blue[i]  >> shift;
      entries[i].pad   = 0;
    }
    err = uvesafb_setpalette(entries, cmap->len, cmap->start, info);
    kfree(entries);
  } else {
    /*
     * For modes with bpp > 8, we only set the pseudo palette in
     * the fb_info struct. We rely on uvesafb_setcolreg to do all
     * sanity checking.
     */
    for (i = 0; i < cmap->len; i++) {
      err |= uvesafb_setcolreg(cmap->start + i, cmap->red[i],
            cmap->green[i], cmap->blue[i],
            0, info);
    }
  }
  return err;
}
'''

text = textwrap.dedent(sample_func)
#print(text)

listObj.append({
  "project": "linux kernel",
  "commit_id": "9f645bcc566a1e9f921bdae7528a01ced5bc3713",
  "target": 0,
  "func": text
})

#print(listObj)

with open(filename, 'w') as json_file:
    json.dump(listObj, json_file, 
                        indent=4,  
                        separators=(',',': '))
 
print('Successfully appended to the JSON file')


