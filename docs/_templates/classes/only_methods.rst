.. role:: hidden

{{ name | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ name }}
    :show-inheritance:

{% if methods %}
.. rubric:: Methods

.. autosummary::
    :toctree:
    :nosignatures:

    {% for item in methods %}
    {%- if not item in inherited_members %}
    {%- if not item.startswith("_") %}
    ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endif %}
    {%- endfor %}
{%- endif %}
