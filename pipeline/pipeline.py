import os
import itertools

import lucidity

class Resource(object):
    pass

class FilePattern(object):

    def __init__(self, root_dir, pattern):
        if os.path.isabs(pattern):
            raise Exception("Patterns should be relative to the root directory.")
        self.root_dir = root_dir
        self.pattern = os.path.join(root_dir, pattern)

        # lucidity requires naming each Template. As a hack for now, we'll just
        # name the template the pattern itself, to ensure there are no
        # conflicts.
        self.template = lucidity.Template(self.pattern, self.pattern)

    def pattern_root(self):

        # Note: this will need to use splitdrive to work on Windows
        components = self.pattern.split('/')
        pattern_root = []
        for component in components:
            if "{" in component:
                break
            pattern_root.append(component)

        return "/".join(pattern_root)

    def discover(self):
        root = self.pattern_root()
        candidates = [os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]
        self.filenames = []
        self.instances = []
        self.variables = None
        for filename in candidates:
            try:
                instance = self.template.parse(filename)

                # Do this before adding the filename to the instance
                if self.variables is None:
                    self.variables = instance.keys()

                instance['filename'] = filename
                self.filenames.append(filename)
                self.instances.append(instance)
            except:
                pass

class ResourceIterator(object):

    def __init__(self, source):
        self.source = source
        self.conditions = {}

    def __iter__(self):
        return self.each()

    def each(self):
        for v in self.source.values():
            if self.accept(v):
                yield v

    def all(self):
        return [list(self.each())]

    def where(self, conditions):
        valid_keys = self.source.valid_keys()
        for k, v in conditions.items():
            if k not in valid_keys:
                message = "Invalid key: {0}. ({1})".format(k, valid_keys)
                raise ValueError(message)
            self.conditions[k] = str(v)
        return self

    def accept(self, item):
        for k, v in self.conditions.items():
            if item[k] != v:
                return False
        return True


class FileResource(Resource):

    def __init__(self, root_dir, pattern):
        self.pattern = FilePattern(root_dir, pattern)
        self.pattern.discover()
        self.variables = self.pattern.variables

    def __iter__(self):
        return ResourceIterator(self).each()

    def each(self):
        return ResourceIterator(self).each()

    def all(self):
        return ResourceIterator(self).all()

    def where(self, **kwargs):
        return ResourceIterator(self).where(kwargs)

    def valid_keys(self):
        return self.variables

    def values(self):
        for instance in self.pattern.instances:
            yield instance


class Pipeline(object):

    def __init__(self):
        self.current_dir = None

    def file(self, pattern):
        if self.current_dir is None:
            raise Exception("Pipeline: must chdir before creating file resource.")
        return FileResource(self.current_dir, pattern)

    def chdir(self, new_dir):
        if new_dir[0] == "/":
            pass
        elif new_dir[0] == ".":
            print "Warning, using relative path based on *where python is invoked*"
            new_dir = os.path.abspath(os.path.expanduser(new_dir))
        else:
            print "Warning, using relative path based on *where python is invoked*"
            new_dir = "./" + new_dir
            new_dir = os.path.abspath(os.path.expanduser(new_dir))

        self.current_dir = new_dir
