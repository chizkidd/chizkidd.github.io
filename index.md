---
layout: default
---



{% for post in site.posts %}
<article class="post-preview">
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <p class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</p>
  <p class="post-excerpt">{{ post.excerpt | truncatewords: 50 }}</p>
</article>
{% endfor %}

