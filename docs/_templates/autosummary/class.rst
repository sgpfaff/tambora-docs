{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :member-order: bysource

   {% block methods %}
   {% set visible = [] %}
   {% for item in methods %}
     {%- if not item.startswith('_') %}{{ visible.append(item) or '' }}{% endif -%}
   {% endfor %}
   {% if visible %}
   .. rubric:: Methods at a glance

   .. autosummary::
      :nosignatures:
   {% for item in visible %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set vattrs = [] %}
   {% for item in attributes %}
     {%- if not item.startswith('_') %}{{ vattrs.append(item) or '' }}{% endif -%}
   {% endfor %}
   {% if vattrs %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in vattrs %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
