import pytest

# from abc import abstractmethod, ABCMeta
from unittest import TestCase

from zipline.utils.final import (
    FinalMeta,
    final,
)

# from zipline.utils.metautils_ import compose_types


class FinalMetaTestCase(TestCase):
    @classmethod
    def setup_class(cls):
        class ClassWithFinal(object, metaclass=FinalMeta):
            a = final("ClassWithFinal: a")
            b = "ClassWithFinal: b"

            @final
            def f(self):
                return "ClassWithFinal: f"

            def g(self):
                return "ClassWithFinal: g"

        cls.class_ = ClassWithFinal

    def test_subclass_no_override(self):
        """
        Tests that it is valid to create a subclass that does not override
        any methods.
        """

        class SubClass(self.class_):
            pass

    def test_subclass_no_final_override(self):
        """
        Tests that it is valid to create a subclass that does not override
        and final methods.
        """

        class SubClass(self.class_):
            b = "SubClass: b"

            def g(self):
                return "SubClass: g"

    def test_override_final_no_decorator(self):
        """
        Tests that attempting to create a subclass that overrides a final
        method will raise a `TypeError`.
        """
        with pytest.raises(TypeError):

            class SubClass(self.class_):
                def f(self):
                    return "SubClass: f"

    def test_override_final_attribute(self):
        """
        Tests that attempting to create a subclass that overrides a final
        attribute will raise a `TypeError`.
        """
        with pytest.raises(TypeError):

            class SubClass(self.class_):
                a = "SubClass: a"

    def test_override_final_with_decorator(self):
        """
        Tests that attempting to create a subclass that overrides a final
        method will raise a `TypeError` even if you mark the new version as
        final.
        """
        with pytest.raises(TypeError):

            class SubClass(self.class_):
                @final
                def f(self):
                    return "SubClass: f"

    def test_override_final_attribute_with_final(self):
        """
        Tests that attempting to create a subclass that overrides a final
        attribute will raise a `TypeError` even if you mark the new version as
        final.
        """
        with pytest.raises(TypeError):

            class SubClass(self.class_):
                a = final("SubClass: a")

    def test_override_on_class_object(self):
        """
        Tests overriding final methods and attributes on the class object
        itself.
        """

        class SubClass(self.class_):
            pass

        with pytest.raises(TypeError):
            SubClass.f = lambda self: "SubClass: f"

        with pytest.raises(TypeError):
            SubClass.a = "SubClass: a"

    def test_override_on_instance(self):
        """
        Tests overriding final methods on instances of a class.
        """

        class SubClass(self.class_):
            def h(self):
                pass

        s = SubClass()
        with pytest.raises(TypeError):
            s.f = lambda self: "SubClass: f"

        with pytest.raises(TypeError):
            s.a = lambda self: "SubClass: a"

    def test_override_on_super(self):
        """
        Tests overriding on the class that has the @final methods in it.
        """
        old_a = self.class_.a
        old_f = self.class_.f
        try:
            with pytest.raises(TypeError):
                self.class_.f = lambda *args: None
        except Exception:
            self.class_.f = old_f
            raise

        try:
            with pytest.raises(TypeError):
                self.class_.a = "SubClass: a"
        except Exception:
            self.class_.a = old_a
            raise

    def test_override___setattr___on_subclass(self):
        """
        Tests an attempt to override __setattr__ which is implicitly final.
        """
        with pytest.raises(TypeError):

            class SubClass(self.class_):
                def __setattr__(self, name, value):
                    object.__setattr__(self, name, value)

    def test_override___setattr___on_instance(self):
        """
        Tests overriding __setattr__ on an instance.
        """

        class SubClass(self.class_):
            pass

        s = SubClass()
        with pytest.raises(TypeError):
            s.__setattr__ = lambda a, b: None


class FinalABCMetaTestCase(FinalMetaTestCase):
    # @classmethod
    # def setup_class(cls):
    #     FinalABCMeta = compose_types(FinalMeta, ABCMeta)
    #
    #     class ABCWithFinal(with_metaclass(FinalABCMeta, object)):
    #         a = final("ABCWithFinal: a")
    #         b = "ABCWithFinal: b"
    #
    #         @final
    #         def f(self):
    #             return "ABCWithFinal: f"
    #
    #         def g(self):
    #             return "ABCWithFinal: g"
    #
    #         @abstractmethod
    #         def h(self):
    #             raise NotImplementedError("h")
    #
    #     cls.class_ = ABCWithFinal
    #
    # def test_cannot_instantiate_subclass(self):
    #     """
    #     Tests that you cannot create an instance of a subclass
    #     that does not implement the abstractmethod h.
    #     """
    #
    #     class AbstractSubClass(self.class_):
    #         pass
    #
    #     with self.assertRaises(TypeError):
    #         AbstractSubClass()
    #
    # def test_override_on_instance(self):
    #     class SubClass(self.class_):
    #         def h(self):
    #             """
    #             Pass the abstract tests by creating this method.
    #             """
    #             pass
    #
    #     s = SubClass()
    #     with self.assertRaises(TypeError):
    #         s.f = lambda self: "SubClass: f"
    #
    # def test_override___setattr___on_instance(self):
    #     """
    #     Tests overriding __setattr__ on an instance.
    #     """
    #
    #     class SubClass(self.class_):
    #         def h(self):
    #             pass
    #
    #     s = SubClass()
    #     with self.assertRaises(TypeError):
    #         s.__setattr__ = lambda a, b: None

    def test_subclass_setattr(self):
        """Tests that subclasses don't destroy the __setattr__."""

        class ClassWithFinal(object, metaclass=FinalMeta):
            @final
            def f(self):
                return "ClassWithFinal: f"

        class SubClass(ClassWithFinal):
            def __init__(self):
                self.a = "a"

        SubClass()
        assert SubClass().a == "a"
        assert SubClass().f() == "ClassWithFinal: f"

    def test_final_classmethod(self):
        class ClassWithClassMethod(object, metaclass=FinalMeta):
            count = 0

            @final
            @classmethod
            def f(cls):
                cls.count += 1
                return cls.count

        with pytest.raises(TypeError):

            class ClassOverridingClassMethod(ClassWithClassMethod):
                @classmethod
                def f(cls):
                    return "Oh Noes!"

        with pytest.raises(TypeError):
            ClassWithClassMethod.f = lambda cls: 0

        assert ClassWithClassMethod.f() == 1
        assert ClassWithClassMethod.f() == 2
        assert ClassWithClassMethod.f() == 3

        instance = ClassWithClassMethod()

        with pytest.raises(TypeError):
            instance.f = lambda cls: 0

        assert ClassWithClassMethod.f() == 4
        assert ClassWithClassMethod.f() == 5
        assert ClassWithClassMethod.f() == 6
