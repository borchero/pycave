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

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
    {% for item in attributes if item in ["model_"] %}
    ~{{ name }}.{{ item }}
    {%- endfor %}
{%- endif %}
