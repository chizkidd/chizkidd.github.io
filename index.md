---
layout: default
---

# Welcome to my blog

{% for post in site.posts %}
* {{ post.date | date: "%b %-d, %Y" }}
  [{{ post.title }}]({{ post.url | relative_url }})
  
  {{ post.excerpt }}
{% endfor %}

subscribe [via RSS]({{ "/feed.xml" | relative_url }})
