---
layout: default
---

{% for post in site.posts %}
<div class="post-header">
  <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
  <h2 class="post-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <p class="post-meta">{{ post.excerpt | truncatewords: 50 }}</p>
</div>
{% endfor %}

<p>subscribe <a href="/feed.xml">via RSS</a></p>


