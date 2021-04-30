def bulleted_list(items, max_count=None, indent=2):
    """Format a bulleted list of values."""
    if max_count is not None and len(items) > max_count:
        item_list = list(items)
        items = item_list[: max_count - 1]
        items.append("...")
        items.append(item_list[-1])

    line_template = (" " * indent) + "- {}"
    return "\n".join(map(line_template.format, items))
