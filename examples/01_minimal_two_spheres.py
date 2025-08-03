"""
01 – Simple example
======================================
...
"""

# This is commented python
myvariable = 2
print("my variable is {}".format(myvariable))

# %%
# This is a section header
# ------------------------
#
# In the built documentation, it will be rendered as reST. All reST lines
# must begin with '# ' (note the space) including underlines below section
# headers.

# These lines won't be rendered as reST because there is a gap after the last
# commented reST block. Instead, they'll resolve as regular Python comments.
# Normal Python code can follow these comments.
print("my variable plus 2 is {}".format(myvariable + 2))
